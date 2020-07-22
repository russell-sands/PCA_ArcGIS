import pandas as pd
import numpy as np
import arcpy
import os
from arcgis.features import GeoAccessor, GeoSeriesAccessor
from sklearn.decomposition import PCA

def readFieldTable(parameterValue):
    '''value table returns strings with single quote and we don't want that'''
    out = []
    for i in range(parameterValue.rowCount - 1):
        rowVal = parameterValue.getRow(i)
        if rowVal[0] == "'" and rowVal[-1] =="'":
            rowVal = rowVal[1:-1]
        out.append(rowVal)
    return out

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [PCA_Result]


class PCA_Result(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Run PCA (scikit-learn)"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        
        # # Template
        # p = arcpy.Parameter(
        #     displayName = "",
        #     name = "",
        #     datatype = "",
        #     parameterType = "",
        #     direction = "")
        
        p0 = arcpy.Parameter(
            displayName = "Input Features",
            name = "srcFeatures",
            datatype = "GPFeatureLayer",
            parameterType = "Required",
            direction = "Input")
        
        p1= arcpy.Parameter(
            displayName = "ID Field",
            name = "srcIdField",
            datatype = "Field",
            parameterType = "Required",
            direction = "Input")
        p1.parameterDependencies = [p0.name]

        p2 = arcpy.Parameter(
            displayName = "Input fields for PCA",
            name = "srcFields",
            datatype = "Field",
            parameterType = "Required",
            direction = "Input",
            multiValue = True)
        p2.parameterDependencies = [p0.name]
        p2.filter.list = ["Short", "Long", "Float", "Double"]

        p3 = arcpy.Parameter(
            displayName = "PCA Components",
            name = "nComponents",
            datatype = "Long",
            parameterType = "Required",
            direction = "Input")

        p4 = arcpy.Parameter(
            displayName = "Output File",
            name = "outFile",
            datatype = "DEFeatureClass",
            parameterType = "Required",
            direction = "Ouput")
        
        params = [p0, p1, p2, p3, p4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):

        # Read in the input data
        srcDesc = arcpy.Describe(parameters[0].valueAsText)
        srcId = parameters[1].valueAsText
        pcaFields = readFieldTable(parameters[2].value)
        nComPca = parameters[3].value
        out = parameters[4].valueAsText

        # Get the path to the data source. Required for reading data in to pd
        src = os.path.join(srcDesc.path, srcDesc.baseName)
        if srcDesc.extension:
            src += "." + srcDesc.extension

        # Run a PCA analysis by reading the data into a pandas dataframe and 
        #   using the PCA analyis available in scikit-learn
        pca = PCA(n_components = nComPca)
        dfSrc = pd.DataFrame.spatial.from_featureclass(src)
        dfPca = dfSrc[pcaFields] #Subset of the dataframe w/ only pca fields
        dfTransformed = pd.DataFrame(pca.fit_transform(dfPca))
        dfTransformedCols = ["PCA_%s" % i for i in range(nComPca)]
        dfTransformed.columns = dfTransformedCols
        dfResult = dfSrc[[srcId, "SHAPE"]]
        dfResult = dfResult.join(dfTransformed)
        dfResult.spatial.to_featureclass(out)
        arcpy.AddMessage("Explained Varience")
        arcpy.AddMessage(pca.explained_variance_ratio_)

        return