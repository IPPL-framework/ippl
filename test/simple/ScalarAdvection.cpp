// ----------------------------------------------------------------------------
// Scalar advection example.
//
// This is here for comparison with IPPL 2.x, where the equivalent test is in
// r2/src/Field/tests.
// ----------------------------------------------------------------------------

// include files

#include "Ippl.h"
#include "Clock.h" // Borrowed from IPPL 2.

#include <iostream>

#include <stdio.h>

// Forward declarations:
template<class T, class M, class C>
void textOut(Field<T,1,M,C> &f);
template<class T, class M, class C>
void textOut(Field<T,2,M,C> &f);
template<class T, class M, class C>
void textOut(Field<T,3,M,C> &f);

// SAOptions - options parsing class.
// Member functions defined after main().

template <int D>
class SAOptions;

template <class OStr>
void usage(const std::string &name, OStr &out);

template <int D, class OStr>
void print(const SAOptions<D> &opts, OStr &out);

template <class OStr>
void badOption(OStr &out, const char *str, const std::string &option);

template <class OStr>
void badValue(OStr &out, const std::string &option);

template <int D>
class SAOptions
{
public:

  double epsilon;           // Purge threshold
  double dt;                // Time step
  bool purge;               // To purge or not to purge
  bool doTextOut;           // Text output of u, every outputIncrement steps
  bool doEnsightOut;        // Ensight output of u, every outputIncrement steps
  bool doSumOut;            // Output sum(u), every outputIncrement steps
  int lastTimeStep;         // Last time step
  int outputIncrement;      // Frequency of optional output
  int purgeIncrement;       // How often to purge
  int nPatches[D];          // Number of patches
  int nCells[D];            // Number of cells
  int nVerts[D];            // Number of verts (calculated from nCells)
  int pulseHalfWidthCells;  // Half-width (in cells) of (symmetrical) loaded
                            // pulse
  std::string progname;     // Name of program.

  // Default constructor, sets the default options.

  SAOptions();

  // Set options from argc, argv.

  SAOptions(int argc, char *argv[]);

  // Prints a summary of the options.

  template <class OStr>
  void print(OStr &out) { ::print(*this, out); }

  // Prints a usage message.

  template <class OStr>
  void usage(OStr &out) { ::usage(progname, out); }

private:

  // Helper functions used in parsing options with int and double arguments:

  static bool intArgument(int argc, char **argv, int pos, int &val);
  static bool doubleArgument(int argc, char **argv, int pos, double &val);

  // Report bad option value
  // (These are forwarded to globals as I don't want to put the
  // bodies inline here. Fix when CW is fixed.)

  template <class OStr>
  static void
  badValue(OStr &out, const std::string &option) { ::badValue(out,option); }

  template <class OStr>
  static void
  badOption(OStr &out, const char *str, const std::string &option)
  { ::badOption(out,str,option); }

};

int main(int argc, char *argv[])
{
  Ippl ippl(argc,argv); // Set up the library
  Inform pout(NULL, 0);   // Output, via process 0 only

  const int Dim = 2; // Set the dimensionality

  // The SAOptions object sets the default option values
  // and parses argv for options that override the defaults.

  SAOptions<Dim> opts(argc, argv);

  opts.print(pout);

  // Create the physical domains:
  NDIndex<Dim> vertexDomain, cellDomain;
  for (int d = 0; d < Dim; d++) {
    vertexDomain[d] = Index(opts.nVerts[d]);
    cellDomain[d] = Index(opts.nVerts[d] - 1);
  }

  // Create the (uniform, logically rectilinear) mesh.
  Vektor<double,Dim> origin(0.0);
  double spacings[Dim];
  for (int d = 0; d < Dim; d++) { spacings[d] = 0.2; }
  typedef UniformCartesian<Dim,double> Mesh_t;
  Mesh_t mesh(vertexDomain, spacings, origin);

  // Create the layouts
  e_dim_tag edt[Dim];
  unsigned nVnodes[Dim];
  for (int d = 0; d < Dim; d++) {
    nVnodes[d] = opts.nPatches[d];
    edt[d] = PARALLEL;
  }
  CenteredFieldLayout<Dim,Mesh_t,Cell> layoutc(mesh, edt, nVnodes, true);

  // Store the mesh spacing fields:
  mesh.storeSpacingFields(edt, nVnodes, true);

  // Set up periodic boundary conditions on all mesh faces:
  BConds<double,Dim> bc;
  BConds<Vektor<double,Dim>,Dim> vbc;
  for (int face = 0; face < 2*Dim; face++) {
    bc[face] = new ParallelPeriodicFace<double,Dim>(face);
    vbc[face] = new ParallelPeriodicFace<Vektor<double,Dim>,Dim>(face);
  }

  // Create the Fields:

  // The flow Field u(x,t), a duplicate (stored at the previous
  // timestep for staggered leapfrog), and a useful temporary:
  GuardCellSizes<Dim> gc(1);
  Field<double, Dim, Mesh_t, Cell> u(mesh, layoutc, gc, bc);
  Field<double, Dim, Mesh_t, Cell> uPrev(mesh, layoutc);
  Field<double, Dim, Mesh_t, Cell> uTemp(mesh, layoutc);
  // Extra scalar and Vektor Field tempoaries needed for r1 Div() usage:
  Field<double, Dim, Mesh_t, Cell> uTemp2(mesh, layoutc);
  Field<Vektor<double,Dim>, Dim, Mesh_t, Cell> uv(mesh, layoutc, gc, vbc);

  // Extra Field to store the cell-center positions:
  // Give it linear-extrapolation BC:
  BConds<Vektor<double,Dim>,Dim> xbc;
  for (int face = 0; face < 2*Dim; face++) {
    xbc[face] = new LinearExtrapolateFace<Vektor<double,Dim>,Dim>(face);
  }
  Field<Vektor<double,Dim>, Dim, Mesh_t, Cell> x(mesh, layoutc, gc, xbc);
  // Assign it to the positions computed by the mesh:
  x = mesh.getCellPositionField(x);

  // Initialize Fields to zero everywhere (note: can't assign global guard
  // layers in r1):
  u = 0.0;

  // Load initial condition u(x,0), a symmetric pulse centered around nCells/4
  // and decaying to zero away from nCells/4 all directions, with a height of
  // 1.0, with a half-width equal to opts.pulseHalfWidthCells (defaults to
  // nCells/8):
  double pulseHalfWidth = spacings[0]*opts.pulseHalfWidthCells;
  NDIndex<Dim> pulseCenter;
  for (int d = 0; d < Dim; d++) {
    pulseCenter[d] = Index(opts.nCells[0]/4,opts.nCells[0]/4);
  }
  Vektor<double,Dim> u0 = x[pulseCenter].get();
  u = 1.0 * exp(-dot(x - u0, x - u0) / (2.0 * pulseHalfWidth));

  // Print out information at t = 0:
  if (opts.purge) {
    u = where(lt(abs(u), opts.epsilon), 0.0); // Purge initial conditions
  }
//   u.fillGuardCells(); // Avoids getting overly high reported compression
//   u.Compress();       // Avoids getting overly low reported compression
  pout << "t = " << double(0.0) << " ; u.CompressedFraction() = "
       << u.CompressedFraction() << endl;
  if (opts.doSumOut) pout << "sum(u) = " << sum(u) << endl;
  if (opts.doTextOut) { setInform(pout); }
  if (opts.doTextOut) { setFormat(6,3,10); }
  if (opts.doTextOut) { textOut(u); }

  const Vektor<double,Dim> v(0.2);   // Propagation velocity
  const double dt = opts.dt;         // Timestep

  // Prime the leapfrog by setting the field at the previous timestep using the
  // initial conditions:
  uPrev = u;

  // Do a preliminary timestep using forward Euler, using Div():
  uv = u * v * dt;
  uTemp2 = Div(uv, uTemp2);
  u -= uTemp2;

  // Start timer to time main loop:
  double startTime = Clock::value();

  // Now use staggered leapfrog (second-order) for the remaining
  // timesteps.    The spatial derivative is just the second-order
  // finite difference in the canned IPPL stencil-based divergence
  // operator Div():
  for (int timestep = 2; timestep <= opts.lastTimeStep; timestep++) {
    uTemp = u;
    uv = u * v * dt;
    uTemp2 = Div(uv, uTemp2);
    u = uPrev - 2.0 * uTemp2;
    if (opts.purge) {
      if ((timestep % opts.purgeIncrement) == 0) {
        u = where(lt(abs(u), opts.epsilon), 0.0);
      }
    }
    if ((timestep % opts.outputIncrement) == 0) {
      pout << "t = " << timestep*dt
           << " ; u.CompressedFraction() = " << u.CompressedFraction()
           << endl;
      if (opts.doSumOut) pout << "sum(u) = " << sum(u) << endl;
      if (opts.doTextOut) { textOut(u); }
    }
    uPrev = uTemp;
  }

  // Compute the wallclock time for the loop; first do blockAndEvaluate() to
  // make sure the calculation has actually been done first:
  double wallTime = Clock::value() - startTime;

  pout << "Done. Wallclock seconds = " << wallTime << endl;
  return 0;
}


template<class T, class M, class C>
void textOut(Field<T,1,M,C> &f) {
  fp1(f);
}
template<class T, class M, class C>
void textOut(Field<T,2,M,C> &f) {
  fp2(f);
}
template<class T, class M, class C>
void textOut(Field<T,3,M,C> &f) {
  fp3(f);
}

// Non-inline function definitions for SAOptions.

template <int D>
SAOptions<D>::
SAOptions()
  : epsilon(1.0e-2),
    dt(.1),
    purge(true),
    doTextOut(false),
    doEnsightOut(false),
    doSumOut(true),
    lastTimeStep(1000),
    outputIncrement(10),
    purgeIncrement(1)
{
  for (int d = 0; d < D; ++d)
    {
      nPatches[d] = 10;
      nCells[d]   = 100;
      nVerts[d]   = nCells[d] + 1;
    }
  pulseHalfWidthCells = nCells[0]/8.0;
}

template <int D>
SAOptions<D>::
SAOptions(int argc, char *argv[])
{
    // Set the defaults (default copy constructor OK)

    *this = SAOptions();
    progname = argv[0];

    // Parse the argument list...

    int i = 1;
    while (i < argc)
      {
        using std::string;

        bool hasarg = false;

        string arg(argv[i]);

        if (arg == "-help")
          {
            usage(std::cerr);
            exit(0);
          }
        else if (arg == "-purge")
          {
            purge = true;

            // Check for optional argument:

            hasarg = intArgument(argc, argv, i+1, purgeIncrement);
            if (hasarg) ++i;
          }
        else if (arg == "-text")
          {
            doTextOut = true;
          }
        else if (arg == "-ensight")
          {
            doEnsightOut = true;
          }
        else if (arg == "-sum")
          {
            doSumOut = true;
          }
        else if (arg == "-steps")
          {
            if (i+1 == argc) badOption(std::cerr,
                                       "No value specified for: ", arg);

            hasarg = intArgument(argc, argv, i+1, lastTimeStep);

            if (!hasarg) badOption(std::cerr,
                                   "No value specified for: ", arg);

            ++i;
          }
        else if (arg == "-out")
          {
            if (i+1 == argc) badOption(std::cerr,
                                       "No value specified for: ", arg);

            hasarg = intArgument(argc, argv, i+1, outputIncrement);

            if (!hasarg) badOption(std::cerr, "No value specified for: ", arg);

            ++i;
          }
        else if (arg == "-cells")
          {
            // This can be followed by either 1 int or D ints.

            bool hasarg = intArgument(argc, argv, i+1, nCells[0]);

            if (hasarg)
              {
                ++i;
                if (D > 1)
                  {
                    bool moreArgs = intArgument(argc, argv, i+1, nCells[1]);
                    if (moreArgs)
                      {
                        for (int d = 1; d < D; ++d)
                          {
                            hasarg = intArgument(argc, argv, i+1, nCells[d]);
                            if (!hasarg)
                              badOption(std::cerr,
                                        "Not enough arguments for: ", arg);
                            ++i;
                          }
                      }
                    else
                      {
                        for (int d = 1; d < D; ++d)
                          {
                            nCells[d] = nCells[0];
                          }
                      }
                  }
              }
            else
              {
                badOption(std::cerr, "No argument specified for: ", arg);
              }
            pulseHalfWidthCells = nCells[0]/8.0;
          }
        else if (arg == "-patches")
          {
            // This can be followed by either 1 int or D ints.

            bool hasarg = intArgument(argc, argv, i+1, nPatches[0]);

            if (hasarg)
              {
                ++i;
                if (D > 1)
                  {
                    bool moreArgs = intArgument(argc, argv, i+1, nPatches[1]);
                    if (moreArgs)
                      {
                        for (int d = 1; d < D; ++d)
                          {
                            hasarg = intArgument(argc, argv, i+1, nPatches[d]);
                            if (!hasarg)
                              badOption(std::cerr,
                                        "Not enough arguments for: ", arg);
                            ++i;
                          }
                      }
                    else
                      {
                        for (int d = 1; d < D; ++d)
                          {
                            nPatches[d] = nPatches[0];
                          }
                      }
                  }
              }
            else
              {
                badOption(std::cerr, "No argument specified for: ", arg);
              }
          }
        else if (arg == "-epsilon")
          {
            if (i+1 == argc) badOption(std::cerr,
                                       "No value specified for: ", arg);

            hasarg = doubleArgument(argc, argv, i+1, epsilon);

            if (!hasarg) badOption(std::cerr, "No value specified for: ", arg);

            ++i;
          }
        else if (arg == "-dt")
          {
            if (i+1 == argc) badOption(std::cerr,
                                       "No value specified for: ", arg);

            hasarg = doubleArgument(argc, argv, i+1, dt);

            if (!hasarg) badOption(std::cerr, "No value specified for: ", arg);

            ++i;
          }
        else if (arg == "-pulseHalfWidthCells")
          {
            if (i+1 == argc) badOption(std::cerr,
                                       "No value specified for: ", arg);

            hasarg = intArgument(argc, argv, i+1, pulseHalfWidthCells);

            if (!hasarg) badOption(std::cerr, "No value specified for: ", arg);

            ++i;
          }
        else
          {
            std::cerr << "No such flag: " << arg << std::endl;
            usage(std::cerr);
            exit(0);
          }

        ++i; // next arg
      }

    // Do some sanity checks:

    if (lastTimeStep < 1)
      badValue(std::cerr, "-steps");
    if (outputIncrement < 1 || outputIncrement > lastTimeStep)
      badValue(std::cerr, "-out");
    if (purgeIncrement < 1 || purgeIncrement > lastTimeStep)
      badValue(std::cerr, "-purge");

    // Finally, initialize nVerts.

    for (int d = 0; d < D; ++d) nVerts[d] = nCells[d] + 1;

}

template <int D>
bool SAOptions<D>::
intArgument(int argc, char **argv, int pos, int &val)
{
    // Make sure there is an argument available

    if (pos >= argc)
      return false;

    // Make sure the 'pos' argument is a number.  If it starts with a number
    // or -number, it is OK.

    char firstchar = argv[pos][0];
    if (firstchar < '0' || firstchar > '9')
      {
        // first char is not a number.  Is the second, with the first a '-/+'?

        if ((firstchar != '-' && firstchar != '+') || argv[pos][1] == 0 ||
            (argv[pos][1] < '0' || argv[pos][1] > '9'))
          return false;
      }

    // Get the value and return it in the last argument

    val = atoi(argv[pos]);
    return true;
}

template <int D>
bool SAOptions<D>::
doubleArgument(int argc, char **argv, int pos, double &val)
{
    // Make sure there is an argument available

    if (pos >= argc)
      return false;

    // Make sure the 'pos' argument is a number.  If it starts with a number
    // or -number, it is OK.

    char firstchar = argv[pos][0];
    if (firstchar < '0' || firstchar > '9')
      {
        // first char is not a number.  Is the second, with the first a '-/+'?

        if ((firstchar != '-' && firstchar != '+') || argv[pos][1] == 0 ||
            (argv[pos][1] < '0' || argv[pos][1] > '9'))
          return false;
      }

    // Get the value and return it in the last argument

    val = atof(argv[pos]);
    return true;
}

//
// Helper functions: print, usage, badOption, badValue
//
// To avoid having to put these functions in the class body, I've written
// them as global template functions. The corresponding member functions
// simply call these.
//

template <int D, class OStr>
void print(const SAOptions<D> &opts, OStr &pout)
{
//     using std::endl;

    int d;

    pout << "Program name: " << opts.progname << endl;
    pout << "Option values: " << endl;
    pout << "=====================================================" << endl;

    pout << "text                = "
         << (opts.doTextOut ? "true ; " : "false ; ") << endl;
    pout << "sum                 = "
         << (opts.doSumOut ? "true ; " : "false ; ") << endl;
    pout << "ensight             = "
         << (opts.doEnsightOut ? "true ; " : "false ; ") << endl;
    pout << "purge               = "
         << (opts.purge ? "true ; " : "false ; ") << endl;

    pout << "time step           = " << opts.dt << endl;
    pout << "steps               = " << opts.lastTimeStep << endl;
    pout << "outSteps            = " << opts.outputIncrement << endl;
    pout << "purgeSteps          = " << opts.purgeIncrement << endl;

    pout << "nCells              = " << opts.nCells[0];
    for (d = 1; d < D; ++d)
      pout << ", " << opts.nCells[d];
    pout << endl;

    pout << "nVerts              = " << opts.nVerts[0];
    for (d = 1; d < D; ++d)
      pout << ", " << opts.nVerts[d];
    pout << endl;

    pout << "nPatches            = " << opts.nPatches[0];
    for (d = 1; d < D; ++d)
      pout << ", " << opts.nPatches[d];
    pout << endl;

    pout << "epsilon             = " << opts.epsilon << endl;
    pout << "pulseHalfWidthCells = " << opts.pulseHalfWidthCells << endl;
    pout << "=====================================================" << endl
         << endl;
}

template <class OStr>
void usage(const std::string &name, OStr &out)
{
    out << "Usage: " << name << std::endl
        << " [-cells <nCellsX> [<nCellsY> <nCellsZ>]]"
        << std::endl
        << " [-patches <nPatchesX> [<nPatchesY> <nPatchesZ>]]"
        << std::endl
        << " [-dt <timestep>]"
        << " [-steps <lastTimeStep>]"
        << std::endl
        << " [-pulseHalfWidthCells <pulseHalfWidthCells>]"
        << std::endl
        << " [-out <outputIncrement>]"
        << " [-sum]"
        << " [-text]"
        << " [-ensight]"
        << std::endl
        << " [-purge [<purgeIncrement>]]"
        << " [-epsilon <epsilon>]"
        << " [-block]"
        << std::endl;
}

template <class OStr>
void badOption(OStr &out, const char *str, const std::string &option)
{
  out << "Bad option: " << str << option << std::endl;
  exit(1);
}

template <class OStr>
void badValue(OStr &out, const std::string &option)
{
  out << "Bad input value for option: " << option << std::endl;
  exit(1);
}