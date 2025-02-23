# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
ippl_field = PVTrivialProducer(registrationName='ippl_field')
ippl_particle = PVTrivialProducer(registrationName='ippl_particle')

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

vTPD1 = CreateExtractor('VTPD', ippl_field, registrationName='VTPD1')
# trace defaults for the extractor.
vTPD1.Trigger = 'TimeStep'

# init the 'VTPD' selected for 'Writer'
vTPD1.Writer.FileName = 'ippl_field_{timestep:06d}.vtpd'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vTPD1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

vTPD2 = CreateExtractor('VTPD', ippl_particle, registrationName='VTPD2')
# trace defaults for the extractor.
vTPD2.Trigger = 'TimeStep'

# init the 'VTPD' selected for 'Writer'
vTPD2.Writer.FileName = 'ippl_particle_{timestep:06d}.vtpd'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vTPD1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GenerateCinemaSpecification = 1
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
