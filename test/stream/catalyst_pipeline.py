from paraview.simple import *

print("executing catalyst_pipeline")



# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")

def catalyst_execute(info):
    global producer
    producer.UpdatePipeline()
    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    #print("field:", producer.CellData["density"])
#    print("pressure-range:", producer.CellData["pressure"].GetRange(0))
