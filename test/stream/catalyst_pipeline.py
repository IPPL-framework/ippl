from paraview.simple import *
from paraview import catalyst
options = catalyst.Options()

print("executing catalyst_pipeline")


# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="ippl_field")
def catalyst_execute(info):
    global producer
    #SaveExtractsUsingCatalystOptions(options)
#    global producer
    producer.UpdatePipeline(info.time)
#    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    arrayInfo = producer.CellData["density"]
    arrayInfo.GetNumberOfComponents()
    print("field:", producer.CellData["density"].GetRange(-1)) # .GetRange(0))
#    print("pressure-range:", producer.CellData["pressure"].GetRange(0))
