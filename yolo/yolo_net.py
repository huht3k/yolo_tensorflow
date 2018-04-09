import numpy as np
import tensorflow as tf
import yolo.config as cfg


class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE            # IMAGE_SIZE = 448
        self.cell_size = cfg.CELL_SIZE              # CELL_SIZE = 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL    # BOXES_PER_CELL = 2
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)      # 49 * 30 = 1470
        self.scale = 1.0 * self.image_size / self.cell_size     # 448/7.0 = 64.0
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  # 7 * 7 * 20 = 980
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell   # 980 + 7 * 7 * 2 = 1078

        self.object_scale = cfg.OBJECT_SCALE    # OBJECT_SCALE = 1.0
        self.noobject_scale = cfg.NOOBJECT_SCALE     # NOOBJECT_SCALE = 1.0
        self.class_scale = cfg.CLASS_SCALE           # CLASS_SCALE = 2.0
        self.coord_scale = cfg.COORD_SCALE           # COORD_SCALE = 5.0

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.reg = cfg.REG

        self.disp_console = True
        self.is_training = is_training
        # self.weights_file = 'weights/YOLO_small.ckpt'

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')

        self.logits = self.build_network(self.images)

        if self.is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      keep_prob=0.5):
        if self.disp_console:
            print("Building YOLO_small graph...")
        self.conv_1 = self.conv_layer(1, images, 64, 7, 2)  # Mr. Hu: 224
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)  # Mr. Hu: 112
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)  # Mr. Hu: 56
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)  # Mr. Hu: 28
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)  # Mr. Hu: 14
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)  # Mr. Hu: 7
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
        self.fc_29 = self.fc_layer(29, self.conv_28, 512, flat=True, linear=False)
        self.fc_30 = self.fc_layer(30, self.fc_29, 4096, flat=False, linear=False)
        if self.is_training:
            self.fc_31 = self.dropout_layer(31, self.fc_30, keep_prob)
            self.fc_32 = self.fc_layer(32, self.fc_31, 1470, flat=False, linear=True)
        else:
            self.fc_32 = self.fc_layer(32, self.fc_30, 1470, flat=False, linear=True)
        return self.fc_32


    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        conv_weight_loss = tf.multiply(tf.nn.l2_loss(weight), self.reg, name='conv_weight_loss_' + str(idx))
        # tf.add_to_collection('losses', weight_loss)
        tf.losses.add_loss(conv_weight_loss)

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        if self.disp_console:
            print('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
            idx, size, size, stride, filters, int(channels)))
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console: print
        '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')


    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))

        fc_weight_loss = tf.multiply(tf.nn.l2_loss(weight), self.reg, name='fc_weight_loss_' + str(idx))
        tf.losses.add_loss(fc_weight_loss)

        if self.disp_console:
            print('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
            idx, hiddens, int(dim), int(flat), 1 - int(linear)))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')

        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')


    def dropout_layer(self, idx, inputs, keep_prob):
        if self.disp_console:
            print('    Layer  %d : Type = Dropout, keep_prob = %f ' % (idx, keep_prob))
        return tf.nn.dropout(inputs, keep_prob=keep_prob, name=str(idx) + '_dropout')

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))


            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            # tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            # tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            # tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            # tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            # tf.summary.histogram('iou', iou_predict_truth)
            # tf.summary.histogram('iou_0', iou_predict_truth[..., 0])
            # tf.summary.histogram('iou_1', iou_predict_truth[..., 1])
