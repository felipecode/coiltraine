MODEL_TYPE = 'coil-icra'
MODEL_CONFIGURATION = {
  'perception': {
    'conv': {
      'channels': [32, 32, 64, 64, 128, 128, 256, 256],
      'kernels': [5, 3, 3, 3, 3, 3, 3, 3],
      'strides': [2, 1, 2, 1, 2, 1, 1, 1],
      'dropouts': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      },
    'fc': {
      'neurons': [512, 512],
      'dropouts': [0.0, 0.0]
      }},
  'measurements':{
    'fc': {
      'neurons': [128, 128],
      'dropouts': [0.0, 0.0]
      }},
  'join': {
    'fc': {
      'neurons': [512],
      'dropouts': [0.0]
      }},
  'speed_branch': {
    'fc': {
      'neurons': [256, 256],
      'dropouts': [0.5, 0.5],
      }},
  'branches':{
    'number_of_branches': 4,
    'fc': {
      'neurons': [256, 256],
      'dropouts': [0.5, 0.5]
      }}
}

