from paraview.simple import *

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
    print("field:", producer.CellData["density"].GetRange(-1))
#    print("pressure-range:", producer.CellData["pressure"].GetRange(0))
