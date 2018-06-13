import copy
import tensorflow as tf

from nets.nasnet import nasnet_utils

arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim

from slim.nets.

def _build_nasnet_base(images,
                       normal_cell,
                       reduction_cell,
                       num_classes,
                       hparams,
                       is_training,
                       stem_type,
                       final_endpoint=None):
  """Constructs a NASNet image model."""

  end_points = {}
  def add_and_check_endpoint(endpoint_name, net):
    end_points[endpoint_name] = net
    return final_endpoint and (endpoint_name == final_endpoint)

  # Find where to place the reduction cells or stride normal cells
  reduction_indices = nasnet_utils.calc_reduction_layers(
      hparams.num_cells, hparams.num_reduction_layers)
  stem_cell = reduction_cell

  if stem_type == 'imagenet':
    stem = lambda: _imagenet_stem(images, hparams, stem_cell)
  elif stem_type == 'cifar':
    stem = lambda: _cifar_stem(images, hparams)
  else:
    raise ValueError('Unknown stem_type: ', stem_type)
  net, cell_outputs = stem()
  if add_and_check_endpoint('Stem', net): return net, end_points

  # Setup for building in the auxiliary head.
  aux_head_cell_idxes = []
  if len(reduction_indices) >= 2:
    aux_head_cell_idxes.append(reduction_indices[1] - 1)

  # Run the cells
  filter_scaling = 1.0
  # true_cell_num accounts for the stem cells
  true_cell_num = 2 if stem_type == 'imagenet' else 0
  for cell_num in range(hparams.num_cells):
    stride = 1
    if hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    if cell_num in reduction_indices:
      filter_scaling *= hparams.filter_scaling_rate
      net = reduction_cell(
          net,
          scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)),
          filter_scaling=filter_scaling,
          stride=2,
          prev_layer=cell_outputs[-2],
          cell_num=true_cell_num)
      if add_and_check_endpoint(
          'Reduction_Cell_{}'.format(reduction_indices.index(cell_num)), net):
        return net, end_points
      true_cell_num += 1
      cell_outputs.append(net)
    if not hparams.skip_reduction_layer_input:
      prev_layer = cell_outputs[-2]
    net = normal_cell(
        net,
        scope='cell_{}'.format(cell_num),
        filter_scaling=filter_scaling,
        stride=stride,
        prev_layer=prev_layer,
        cell_num=true_cell_num)

    if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
      return net, end_points
    true_cell_num += 1
    if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and
        num_classes and is_training):
      aux_net = tf.nn.relu(net)
      _build_aux_head(aux_net, end_points, num_classes, hparams,
                      scope='aux_{}'.format(cell_num))
    cell_outputs.append(net)

  # Final softmax layer
  with tf.variable_scope('final_layer'):
    net = tf.nn.relu(net)
    net = nasnet_utils.global_avg_pool(net)
    if add_and_check_endpoint('global_pool', net) or not num_classes:
      return net, end_points
    net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
    logits = slim.fully_connected(net, num_classes)

    if add_and_check_endpoint('Logits', logits):
      return net, end_points

    predictions = tf.nn.softmax(logits, name='predictions')
    if add_and_check_endpoint('Predictions', predictions):
      return net, end_points
  return logits, end_points