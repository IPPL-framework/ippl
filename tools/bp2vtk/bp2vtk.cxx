// -----------------------------------------------------------------------------
// bp2vtk — convert an AdiosCatalyst BP (flattened Conduit "multimesh" tree) into
// standard VTK XML files that ParaView can open directly.
//
//   Usage: bp2vtk <input.bp> [output_dir]   (default output_dir = <input>_vtk)
//
// AdiosCatalyst does not write a VTK/Fides/VTX schema, so ParaView's ADIOS2
// readers cannot open its output. This tool reads the raw ADIOS variables
// (catalyst/channels/<ch>/data/block_*/...) via the ADIOS2 C++ API and rebuilds:
//   * uniform field blocks  -> one global vtkImageData (.vti) per step (cell data),
//                              per-rank subdomains placed via their BlocksInfo origin
//   * the particle block    -> one vtkUnstructuredGrid (.vtu) per step (point data)
// plus a .pvd time series per field / for the particles.
//
// Fields carry a distributed decomposition: each rank writes its own subdomain as
// an ADIOS block with its own origin/dims (recovered through BlocksInfo), and the
// field values are the per-rank cell arrays concatenated in the global array. We
// scatter each block back into the correct place in the global image.
// -----------------------------------------------------------------------------
#include <mpi.h>
#include <adios2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{
// ---- generic typed readers, all converted to double for uniform handling ------

template <class T>
std::vector<double> readArrayImpl(adios2::IO& io, adios2::Engine& eng, const std::string& name,
  std::size_t step, std::vector<std::pair<std::size_t, std::size_t>>* blocks)
{
  auto var = io.InquireVariable<T>(name);
  if (!var)
  {
    return {};
  }
  var.SetStepSelection({ step, 1 });
  if (blocks)
  {
    blocks->clear();
    for (const auto& b : eng.BlocksInfo(var, step))
    {
      blocks->emplace_back(b.Start.empty() ? 0 : b.Start[0], b.Count.empty() ? 0 : b.Count[0]);
    }
  }
  std::vector<T> buf;
  eng.Get(var, buf, adios2::Mode::Sync);
  return std::vector<double>(buf.begin(), buf.end());
}

template <class T>
std::vector<double> readScalarBlocksImpl(
  adios2::IO& io, adios2::Engine& eng, const std::string& name, std::size_t step)
{
  auto var = io.InquireVariable<T>(name);
  if (!var)
  {
    return {};
  }
  std::vector<double> out;
  for (const auto& b : eng.BlocksInfo(var, step))
  {
    out.push_back(static_cast<double>(b.Value));
  }
  return out;
}

// Dispatch on the stored ADIOS type. Returns the whole global array for a step
// (all rank blocks concatenated), optionally with each block's [start,count).
std::vector<double> readArray(adios2::IO& io, adios2::Engine& eng, const std::string& name,
  std::size_t step, std::vector<std::pair<std::size_t, std::size_t>>* blocks = nullptr)
{
  const std::string t = io.VariableType(name);
  if (t == "double") return readArrayImpl<double>(io, eng, name, step, blocks);
  if (t == "float") return readArrayImpl<float>(io, eng, name, step, blocks);
  if (t == "int32_t") return readArrayImpl<std::int32_t>(io, eng, name, step, blocks);
  if (t == "int64_t") return readArrayImpl<std::int64_t>(io, eng, name, step, blocks);
  if (t == "uint32_t") return readArrayImpl<std::uint32_t>(io, eng, name, step, blocks);
  if (t == "uint64_t") return readArrayImpl<std::uint64_t>(io, eng, name, step, blocks);
  return {};
}

// Per-rank block scalar values (e.g. origin/x, dims/i), one entry per writer block.
std::vector<double> readScalarBlocks(
  adios2::IO& io, adios2::Engine& eng, const std::string& name, std::size_t step)
{
  const std::string t = io.VariableType(name);
  if (t == "double") return readScalarBlocksImpl<double>(io, eng, name, step);
  if (t == "float") return readScalarBlocksImpl<float>(io, eng, name, step);
  if (t == "int32_t") return readScalarBlocksImpl<std::int32_t>(io, eng, name, step);
  if (t == "int64_t") return readScalarBlocksImpl<std::int64_t>(io, eng, name, step);
  if (t == "uint32_t") return readScalarBlocksImpl<std::uint32_t>(io, eng, name, step);
  if (t == "uint64_t") return readScalarBlocksImpl<std::uint64_t>(io, eng, name, step);
  return {};
}

double firstOr(const std::vector<double>& v, double dflt)
{
  return v.empty() ? dflt : v.front();
}

void writeDataArray(std::ofstream& os, const std::string& name, const std::vector<double>& data,
  int ncomp, bool asInt = false)
{
  os << "        <DataArray type=\"" << (asInt ? "Int64" : "Float64") << "\" Name=\"" << name
     << "\" NumberOfComponents=\"" << ncomp << "\" format=\"ascii\">\n          ";
  os << std::setprecision(17);
  for (std::size_t i = 0; i < data.size(); ++i)
  {
    if (asInt)
      os << static_cast<long long>(std::llround(data[i]));
    else
      os << data[i];
    os << ((i + 1 < data.size()) ? ' ' : '\n');
  }
  os << "        </DataArray>\n";
}

struct FieldInfo
{
  std::string name;
  bool isVector = false;
  bool isInt = false;
};

std::string segBefore(const std::string& s, char c)
{
  auto p = s.find(c);
  return (p == std::string::npos) ? s : s.substr(0, p);
}
} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  if (argc < 2)
  {
    std::cerr << "Usage: bp2vtk <input.bp> [output_dir]\n";
    MPI_Finalize();
    return 1;
  }
  const std::string input = argv[1];
  std::string outdir = (argc > 2) ? argv[2] : (input + "_vtk");
  // strip trailing slash from the .bp for a nicer default name
  if (argc <= 2 && !outdir.empty() && outdir.back() == '/')
  {
    outdir.pop_back();
  }

  adios2::ADIOS adios(MPI_COMM_WORLD);
  adios2::IO io = adios.DeclareIO("bp2vtk");
  io.SetEngine("BP5");
  adios2::Engine eng = io.Open(input, adios2::Mode::ReadRandomAccess);
  const std::size_t nsteps = eng.Steps();

  const auto available = io.AvailableVariables();

  // Discover the (single) channel and its data blocks.
  const std::string chRoot = "catalyst/channels/";
  std::string channel;
  for (const auto& kv : available)
  {
    if (kv.first.rfind(chRoot, 0) == 0)
    {
      channel = segBefore(kv.first.substr(chRoot.size()), '/');
      break;
    }
  }
  if (channel.empty())
  {
    std::cerr << "No catalyst/channels/* variables found in " << input << "\n";
    eng.Close();
    MPI_Finalize();
    return 2;
  }
  const std::string dataBase = chRoot + channel + "/data/";

  std::set<std::string> blocks;
  for (const auto& kv : available)
  {
    if (kv.first.rfind(dataBase, 0) == 0)
    {
      blocks.insert(segBefore(kv.first.substr(dataBase.size()), '/'));
    }
  }

  std::cout << "Channel: " << channel << "  |  steps: " << nsteps << "  |  blocks:\n";
  for (const auto& b : blocks)
  {
    std::cout << "  - " << b << "\n";
  }

  // system("mkdir -p") equivalents via std::filesystem would need C++17 <filesystem>;
  // keep it dependency-light and let the shell create dirs before running. We only
  // open files, so create directories up front here:
  std::string mk = "mkdir -p '" + outdir + "'";
  if (std::system(mk.c_str()) != 0)
  {
    std::cerr << "Could not create output dir " << outdir << "\n";
  }

  // Time values (per step), read from catalyst/state/time if present.
  std::vector<double> times(nsteps);
  const bool haveTime = available.count("catalyst/state/time") > 0;
  for (std::size_t s = 0; s < nsteps; ++s)
  {
    if (haveTime)
    {
      auto v = readScalarBlocks(io, eng, "catalyst/state/time", s);
      times[s] = firstOr(v, static_cast<double>(s));
    }
    else
    {
      times[s] = static_cast<double>(s);
    }
  }

  auto writePVD = [&](const std::string& pvdPath, const std::string& fileFmt) {
    std::ofstream os(pvdPath);
    os << "<?xml version=\"1.0\"?>\n";
    os << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    os << "  <Collection>\n";
    for (std::size_t s = 0; s < nsteps; ++s)
    {
      char buf[512];
      std::snprintf(buf, sizeof(buf), fileFmt.c_str(), static_cast<int>(s));
      os << "    <DataSet timestep=\"" << std::setprecision(17) << times[s] << "\" file=\"" << buf
         << "\"/>\n";
    }
    os << "  </Collection>\n</VTKFile>\n";
  };

  // ---------------------------------------------------------------------------
  for (const auto& block : blocks)
  {
    const std::string bBase = dataBase + block + "/";
    const bool isField = available.count(bBase + "coordsets/cart_uniform_coords/spacing/dx") > 0;
    const bool isParticle = available.count(bBase + "coordsets/p_explicit_coords/values/x") > 0;

    if (isField)
    {
      // Discover field names under this block.
      const std::string fp = bBase + "fields/";
      std::map<std::string, FieldInfo> fields;
      for (const auto& kv : available)
      {
        const std::string& n = kv.first;
        if (n.rfind(fp, 0) != 0)
        {
          continue;
        }
        const std::string rest = n.substr(fp.size()); // e.g. density/values  or  E/values/x
        const std::string fname = segBefore(rest, '/');
        const std::string tail = rest.substr(fname.size());
        FieldInfo fi;
        fi.name = fname;
        if (tail == "/values")
        {
          fi.isVector = false;
        }
        else if (tail == "/values/x")
        {
          fi.isVector = true;
        }
        else
        {
          continue; // association/topology/volume_dependent strings, or y/z comps
        }
        const std::string t = io.VariableType(fi.isVector ? (n) : n);
        fi.isInt = (t.find("int") != std::string::npos);
        fields[fname] = fi;
      }
      if (fields.empty())
      {
        continue;
      }

      const std::string safe = block;
      const std::string subdir = outdir + "/" + safe;
      std::system(("mkdir -p '" + subdir + "'").c_str());

      for (std::size_t s = 0; s < nsteps; ++s)
      {
        // Per-rank geometry.
        auto ox = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/origin/x", s);
        auto oy = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/origin/y", s);
        auto oz = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/origin/z", s);
        auto di = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/dims/i", s);
        auto dj = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/dims/j", s);
        auto dk = readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/dims/k", s);
        const double sx =
          firstOr(readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/spacing/dx", s), 1.0);
        const double sy =
          firstOr(readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/spacing/dy", s), 1.0);
        const double sz =
          firstOr(readScalarBlocks(io, eng, bBase + "coordsets/cart_uniform_coords/spacing/dz", s), 1.0);

        const std::size_t nb = ox.size();
        if (nb == 0)
        {
          continue;
        }
        const double gox = *std::min_element(ox.begin(), ox.end());
        const double goy = *std::min_element(oy.begin(), oy.end());
        const double goz = *std::min_element(oz.begin(), oz.end());

        // Global cell dims from per-rank placement.
        long gcx = 0, gcy = 0, gcz = 0;
        std::vector<long> iob(nb), job(nb), kob(nb), cnx(nb), cny(nb), cnz(nb);
        for (std::size_t b = 0; b < nb; ++b)
        {
          cnx[b] = static_cast<long>(di[b]) - 1;
          cny[b] = static_cast<long>(dj[b]) - 1;
          cnz[b] = static_cast<long>(dk[b]) - 1;
          iob[b] = std::lround((ox[b] - gox) / sx);
          job[b] = std::lround((oy[b] - goy) / sy);
          kob[b] = std::lround((oz[b] - goz) / sz);
          gcx = std::max(gcx, iob[b] + cnx[b]);
          gcy = std::max(gcy, job[b] + cny[b]);
          gcz = std::max(gcz, kob[b] + cnz[b]);
        }
        const std::size_t gcells = static_cast<std::size_t>(gcx) * gcy * gcz;

        // Assemble each field into a global cell array (or 3 for vectors).
        struct OutField
        {
          std::string name;
          int ncomp;
          bool isInt;
          std::vector<double> data;
        };
        std::vector<OutField> outs;

        auto scatterComp = [&](const std::vector<double>& global,
                             const std::vector<std::pair<std::size_t, std::size_t>>& bl,
                             std::vector<double>& dst, int comp, int ncomp) {
          for (std::size_t b = 0; b < nb && b < bl.size(); ++b)
          {
            const std::size_t start = bl[b].first;
            for (long k = 0; k < cnz[b]; ++k)
              for (long j = 0; j < cny[b]; ++j)
                for (long i = 0; i < cnx[b]; ++i)
                {
                  const std::size_t li =
                    static_cast<std::size_t>(i + cnx[b] * (j + cny[b] * k));
                  const std::size_t gi = static_cast<std::size_t>((iob[b] + i) +
                    gcx * ((job[b] + j) + gcy * (kob[b] + k)));
                  if (start + li < global.size() && gi < dst.size() / ncomp)
                  {
                    dst[gi * ncomp + comp] = global[start + li];
                  }
                }
          }
        };

        for (const auto& kv : fields)
        {
          const FieldInfo& fi = kv.second;
          OutField of;
          of.name = fi.name;
          of.isInt = fi.isInt;
          of.ncomp = fi.isVector ? 3 : 1;
          of.data.assign(gcells * of.ncomp, 0.0);
          if (fi.isVector)
          {
            const char* comps[3] = { "x", "y", "z" };
            for (int c = 0; c < 3; ++c)
            {
              std::vector<std::pair<std::size_t, std::size_t>> bl;
              auto g = readArray(io, eng, fp + fi.name + "/values/" + comps[c], s, &bl);
              scatterComp(g, bl, of.data, c, 3);
            }
          }
          else
          {
            std::vector<std::pair<std::size_t, std::size_t>> bl;
            auto g = readArray(io, eng, fp + fi.name + "/values", s, &bl);
            scatterComp(g, bl, of.data, 0, 1);
          }
          outs.push_back(std::move(of));
        }

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/%s_%04d.vti", subdir.c_str(), safe.c_str(),
          static_cast<int>(s));
        std::ofstream os(fname);
        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        os << "  <ImageData WholeExtent=\"0 " << gcx << " 0 " << gcy << " 0 " << gcz << "\" Origin=\""
           << std::setprecision(17) << gox << " " << goy << " " << goz << "\" Spacing=\"" << sx << " "
           << sy << " " << sz << "\">\n";
        os << "    <Piece Extent=\"0 " << gcx << " 0 " << gcy << " 0 " << gcz << "\">\n";
        os << "      <CellData>\n";
        for (const auto& of : outs)
        {
          writeDataArray(os, of.name, of.data, of.ncomp, of.isInt);
        }
        os << "      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
      }

      writePVD(outdir + "/" + safe + ".pvd", safe + "/" + safe + "_%04d.vti");
      std::cout << "  wrote field series: " << safe << ".pvd\n";
    }
    else if (isParticle)
    {
      const std::string cp = bBase + "coordsets/p_explicit_coords/values/";
      const std::string fp = bBase + "fields/";

      // Discover point-data fields (skip the built-in coordinate mirror 'position' duplicate? keep all).
      std::map<std::string, FieldInfo> fields;
      for (const auto& kv : available)
      {
        const std::string& n = kv.first;
        if (n.rfind(fp, 0) != 0)
        {
          continue;
        }
        const std::string rest = n.substr(fp.size());
        const std::string fname = segBefore(rest, '/');
        const std::string tail = rest.substr(fname.size());
        FieldInfo fi;
        fi.name = fname;
        if (tail == "/values")
          fi.isVector = false;
        else if (tail == "/values/x")
          fi.isVector = true;
        else
          continue;
        fi.isInt = (io.VariableType(n).find("int") != std::string::npos);
        fields[fname] = fi;
      }

      const std::string subdir = outdir + "/" + block;
      std::system(("mkdir -p '" + subdir + "'").c_str());

      for (std::size_t s = 0; s < nsteps; ++s)
      {
        auto x = readArray(io, eng, cp + "x", s);
        auto y = readArray(io, eng, cp + "y", s);
        auto z = readArray(io, eng, cp + "z", s);
        const std::size_t np = x.size();

        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/%s_%04d.vtu", subdir.c_str(), block.c_str(),
          static_cast<int>(s));
        std::ofstream os(fname);
        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        os << "  <UnstructuredGrid>\n";
        os << "    <Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\"" << np << "\">\n";

        os << "      <Points>\n";
        os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
        os << std::setprecision(17);
        for (std::size_t i = 0; i < np; ++i)
        {
          os << x[i] << ' ' << (i < y.size() ? y[i] : 0.0) << ' ' << (i < z.size() ? z[i] : 0.0)
             << ((i + 1 < np) ? ' ' : '\n');
        }
        os << "        </DataArray>\n      </Points>\n";

        os << "      <PointData>\n";
        for (const auto& kv : fields)
        {
          const FieldInfo& fi = kv.second;
          if (fi.isVector)
          {
            auto vx = readArray(io, eng, fp + fi.name + "/values/x", s);
            auto vy = readArray(io, eng, fp + fi.name + "/values/y", s);
            auto vz = readArray(io, eng, fp + fi.name + "/values/z", s);
            std::vector<double> inter(np * 3, 0.0);
            for (std::size_t i = 0; i < np; ++i)
            {
              inter[3 * i + 0] = i < vx.size() ? vx[i] : 0.0;
              inter[3 * i + 1] = i < vy.size() ? vy[i] : 0.0;
              inter[3 * i + 2] = i < vz.size() ? vz[i] : 0.0;
            }
            writeDataArray(os, fi.name, inter, 3, false);
          }
          else
          {
            auto v = readArray(io, eng, fp + fi.name + "/values", s);
            writeDataArray(os, fi.name, v, 1, fi.isInt);
          }
        }
        os << "      </PointData>\n";

        // One VTK_VERTEX cell per point so ParaView filters that need cells work.
        os << "      <Cells>\n";
        os << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ";
        for (std::size_t i = 0; i < np; ++i)
          os << i << ((i + 1 < np) ? ' ' : '\n');
        os << "        </DataArray>\n";
        os << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ";
        for (std::size_t i = 0; i < np; ++i)
          os << (i + 1) << ((i + 1 < np) ? ' ' : '\n');
        os << "        </DataArray>\n";
        os << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ";
        for (std::size_t i = 0; i < np; ++i)
          os << "1" << ((i + 1 < np) ? ' ' : '\n');
        os << "        </DataArray>\n";
        os << "      </Cells>\n";

        os << "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
      }

      writePVD(outdir + "/" + block + ".pvd", block + "/" + block + "_%04d.vtu");
      std::cout << "  wrote particle series: " << block << ".pvd\n";
    }
    else
    {
      std::cout << "  (skipping helper block " << block << ")\n";
    }
  }

  eng.Close();
  std::cout << "Done. Open the .pvd files in " << outdir << " with ParaView.\n";
  MPI_Finalize();
  return 0;
}
