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

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
cSV1 = CreateExtractor('CSV', ippl_field, registrationName='CSV1')
# trace defaults for the extractor.
cSV1.Trigger = 'TimeStep'

# init the 'CSV' selected for 'Writer'
cSV1.Writer.FieldAssociation = 'Cell Data'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(cSV1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
