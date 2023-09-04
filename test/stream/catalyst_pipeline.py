from paraview.simple import *

from paraview.catalyst import get_args
# Greeting to ensure that ctest knows this script is being imported
print("executing catalyst_pipeline")
print("executing catalyst_pipeline")
print("===================================")
print("pipeline args={}".format(get_args()))
print("===================================")


# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer()

def catalyst_execute(info):
    global producer
    producer.UpdatePipeline()
    print("-----------------------------------")
    print("executing (cycle={}, time={})".format(info.cycle, info.time))
    print("getcelldatainformation", producer.GetCellDataInformation())
    print("getdatainformation", producer.GetDataInformation())
    print("getpoint data information", producer.GetPointDataInformation())
    print("getpoint data information", producer.ListProperties())
    print("field:", producer.CellData["density"])
#    print("pressure-range:", producer.CellData["pressure"].GetRange(0))
