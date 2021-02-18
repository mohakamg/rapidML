import pandas
from onnxconverter_common import Int64TensorType, FloatTensorType, StringTensorType, BooleanTensorType


def convert_dataframe_schema(dataframe: pandas.DataFrame):
    """
    This function converts the dataframe input into a schema such that
    onnx can use the schema.
    :param dataframe: DataFrame
    :return inputs: List containing the schema as per ONNX
    """
    inputs = []
    for feature, datatype in zip(dataframe.columns, dataframe.dtypes):
        if datatype == 'int64':
            tensor = Int64TensorType([None, 1])
        elif datatype == 'float64':
            tensor = FloatTensorType([None, 1])
        elif datatype == 'bool':
            tensor = BooleanTensorType([None, 1])
        else:
            tensor = StringTensorType([None, 1])
        inputs.append((feature, tensor))
    return inputs
