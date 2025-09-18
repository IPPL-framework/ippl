from paraview.simple import CreateExtractor
# create extractor (PD=partitioned dataset...)
def create_VTPD_extractor(name, object, fr = 10):

    # create extractor (PD=partitioned dataset...)
    vTPD = CreateExtractor('VTPD', object, registrationName='VTPD_'+ name)
    # vTPD2.Trigger = 'TimeStep'  """ not needed"""
    vTPD.Trigger.Frequency = 10
    vTPD.Writer.FileName = 'ippl_'+name+'_{timestep:06d}.vtpd'
    return vTPD
    

    # Alternative: If you want to extract individual partitions as separate files,
    # you could use PVD format:
    # vPVD_field = CreateExtractor('PVD', ippl_field, registrationName='PVD_field')
    # vPVD_field.Trigger = 'Time Step'
    # vPVD_field.Writer.FileName = 'ippl_field_{timestep:06d}.pvd'

    # not working:
    # create extractor (VTU, U = unstructured)
    # create extractor (VTI, I = Image Data for regular Grids ...) 



