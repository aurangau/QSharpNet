import tensorflow as tf

def compute_Q_TF(x, windowSize=8, thres=0.001):
  # Splitting image into patches
  x_p = tf.image.extract_patches(x, (1,windowSize,windowSize,1), (1,windowSize,windowSize,1),rates=[1, 1, 1, 1], padding='VALID')
  x_p_shape = tf.shape(x_p)
  x_p = tf.reshape(x_p, (-1, x_p_shape[1]*x_p_shape[2], windowSize, windowSize, 1))

  # Getting number of patches
  numPatches = tf.shape(x_p)[1]
  x_p = tf.reshape(x_p, (-1, windowSize, windowSize, 1))

  # Getting x and y gradients
  x_sobel = tf.image.sobel_edges(x_p)
  x_sobel = tf.reshape(x_sobel, (-1, numPatches, windowSize, windowSize, 1, 2))
  x_sobel = x_sobel/windowSize

  x_grads = x_sobel[:,:,:,:,:,0]
  y_grads = x_sobel[:,:,:,:,:,1]

  x_grads = tf.reshape(x_grads, (x_p_shape[0], numPatches, -1))
  y_grads = tf.reshape(y_grads, (x_p_shape[0], numPatches, -1))

  _grads = tf.stack([x_grads, y_grads], -1)
  _grads_trans = tf.transpose(_grads, [0, 1, 3, 2])
  _g_t_g = tf.matmul(_grads_trans, _grads)

  _singValues = tf.linalg.svd(_g_t_g, False, False)

  _s1 = _singValues[:,:,0]
  _s2 = _singValues[:,:,1]

  R = (_s1 - _s2)/((_s1 + _s2) + 1e-10)

  thres = tf.sqrt((1 - tf.pow(thres,1/((windowSize*windowSize) - 1)))/(2))
  R_thres = tf.where(R > thres, R, 0)

  Q = R_thres * _s1
  Q = tf.reduce_mean(Q, (-1, ))
  
  return Q

def compute_Q_TF_withCD(x, windowSize=8, thres=0.001):
  # Splitting image into patches
  x_p = tf.image.extract_patches(x, (1,windowSize,windowSize,1), (1,windowSize,windowSize,1),rates=[1, 1, 1, 1], padding='VALID')
  x_p_shape = tf.shape(x_p)
  x_p = tf.reshape(x_p, (-1, x_p_shape[1]*x_p_shape[2], windowSize, windowSize, 1))

  # Getting number of patches
  numPatches = tf.shape(x_p)[1]
  x_p = tf.reshape(x_p, (-1, windowSize, windowSize, 1))

  # x_grads = (x_p[:, :, 2:, :] - x_p[:, :, :-2, :]) / 2.0
  # y_grads = (x_p[:, 2:, :, :] - x_p[:, :-2, :, :]) / 2.0
  x_grads = (x_p[:, :, 2:, :] - x_p[:, :, :-2, :]) / 2.0
  y_grads = (x_p[:, 2:, :, :] - x_p[:, :-2, :, :]) / 2.0

  x_grads = tf.reshape(x_grads, (x_p_shape[0], numPatches, -1))
  y_grads = tf.reshape(y_grads, (x_p_shape[0], numPatches, -1))

  _grads = tf.stack([x_grads, y_grads], -1)
  _grads_trans = tf.transpose(_grads, [0, 1, 3, 2])
  _g_t_g = tf.matmul(_grads_trans, _grads)

  _singValues = tf.linalg.svd(_g_t_g, False, False)

  _s1 = _singValues[:,:,0]
  _s2 = _singValues[:,:,1]

  R = (_s1 - _s2)/((_s1 + _s2) + 1e-10)

  thres = tf.sqrt((1 - tf.pow(thres,1/((windowSize*windowSize) - 1)))/(2))
  R_thres = tf.where(R > thres, R, 0)

  Q = R_thres * _s1
  Q = tf.reduce_mean(Q, (-1, ))
  
  return Q