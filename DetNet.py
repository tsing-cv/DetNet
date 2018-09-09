###################################################
# tsing
# 2018-09-01
###################################################

# -*- coding: utf-8 -*-
import collections # 原生的collections库
import tensorflow as tf
slim = tf.contrib.slim 

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):

def subsample(inputs, factor, scope=None): 
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride=None, rate=None, scope=None): 
    """
    Args:
        inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
        num_outputs: An integer, the number of output filters.
        kernel_size: An int with the kernel_size of the filters.
        stride: An integer, the output stride.
        rate: An integer, rate for atrous convolution.
        scope: Scope.

    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride:
        if stride == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                            padding='SAME', scope=scope)
        else: 
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs, 
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

            return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                            padding='VALID', scope=scope)
    if rate:
        return slim.conv2d(inputs, num_outputs, kernel_size, rate=rate,
                padding='SAME', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    for block in blocks: 
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if isinstance(unit[-1], int):
                        unit_depth, unit_depth_bottleneck, unit_stride = unit
                        net = block.unit_fn(net, 
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=unit_stride)

                    else:
                        unit_depth, unit_depth_bottleneck, bottleneck_class = unit
                        net = block.unit_fn(net, 
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            bottleneck_class=bottleneck_class)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net) 
    return net 

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001, 
                     batch_norm_decay=0.997, 
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    batch_norm_params = { 
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay), 
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc: 
                return arg_sc 


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc: 
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) 
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact') 

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')

        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                    normalizer_fn=None, activation_fn=None,
                                    scope='shortcut')
            
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
        output = shortcut + residual 
        return slim.utils.collect_named_outputs(outputs_collections, 
                                                sc.name,
                                                output)


@slim.add_arg_scope
def bottleneck_detnet(inputs, depth, depth_bottleneck, 
                      bottleneck_class='A',
                      outputs_collections=None, scope=None):
    assert depth == 4 * depth_bottleneck, 'output_channel != 4xinput_channel'
    with tf.variable_scope(scope, 'bottleneck_{}'.format(bottleneck_class), [inputs]) as sc: 
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) 
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact') 

        if bottleneck_class == 'A':
            shortcut = subsample(inputs, 1, 'shortcut')
            
        elif bottleneck_class == 'B':
            shortcut = slim.conv2d(preact, depth, [1, 1], rate=2,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                              scope='conv1')

        residual = conv2d_same(residual, depth_bottleneck, 3, rate=2,
                               scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
        output = shortcut + residual 
        return slim.utils.collect_named_outputs(outputs_collections, 
                                                sc.name,
                                                output)


def detnet(inputs, 
           blocks, 
           num_classes=None, 
           global_pool=True, 
           include_root_block=True, 
           reuse=None, 
           scope=None): 
    with tf.variable_scope(scope, 'detnet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points' 
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection): 
            net = inputs
            if include_root_block: 
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, 
                                    normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1') 
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1') 
            net = stack_blocks_dense(net, blocks) 
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool: 
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) 
            if num_classes is not None:  
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, 
                            normalizer_fn=None, scope='logits') 
            end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions') 
            return net, end_points



def detnet59(inputs,
             num_classes=None,
             global_pool=True,
             reuse=None,
             scope='detnet59'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck_detnet, [(1024, 256, 'B')] + [(1024, 256, 'A')] * 2),
        Block('block5', bottleneck_detnet, [(1024, 256, 'B')] + [(1024, 256, 'A')] * 2),
    ]
    return detnet(inputs, blocks, num_classes, global_pool,
                  include_root_block=True, reuse=reuse, scope=scope)
