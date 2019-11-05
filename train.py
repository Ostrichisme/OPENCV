#========== packages ==========#
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data  # for data loading
import time
import matplotlib.pyplot as plt
#========== paths ==========#
mnist_data_path = 'MNIST_data/'
model_path = 'model/model.ckpt'


#========== functions ==========#
'''generation of weight'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

'''generation of bias'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

'''construction of leNet-5 model'''
def lenet_5_forward_propagation(X):
    """
    @note: construction of leNet-5 forward computation graph:
        CONV1 -> MAXPOOL1 -> CONV2 -> MAXPOOL2 -> FC3 -> FC4 -> SOFTMAX
        
    @param X: input dataset placeholder, of shape (number of examples (m), input size)
    
    @return: A_l, the output of the softmax layer, of shape (number of examples, output size)
    """
    
    # reshape imput as [number of examples (m), weight, height, channel]
    X_ = tf.reshape(X, [-1, 28, 28, 1])  # num_channel = 1 (gray image)
    
    ### CONV1 (f = 5*5*1, n_f = 6, s = 1, p = 'same')
    W_conv1 = weight_variable(shape = [5, 5, 1, 6])
    b_conv1 = bias_variable(shape = [6])
    # shape of A_conv1 ~ [m,28,28,6]
    A_conv1 = tf.nn.relu(tf.nn.conv2d(X_, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)
    
    ### MAXPOOL1 (f = 2*2*1, s = 2, p = 'same')
    # shape of A_pool1 ~ [m,14,14,6]
    A_pool1 = tf.nn.max_pool(A_conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
    
    ### CONV2 (f = 5*5*1, n_f = 16, s = 1, p = 'same')
    W_conv2 = weight_variable(shape = [5, 5, 6, 16])
    b_conv2 = bias_variable(shape = [16])    
    # shape of A_conv2 ~ [m,10,10,16]
    A_conv2 = tf.nn.relu(tf.nn.conv2d(A_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv2)    
    
    ### MAXPOOL2 (f = 2*2*1, s = 2, p = 'same')  
    # shape of A_pool2~ [m,5,5,16]
    A_pool2 = tf.nn.max_pool(A_conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

    ### FC3 (n = 120)
    # flatten the volumn to vector
    A_pool2_flat = tf.reshape(A_pool2, [-1, 5*5*16])
    
    W_fc3 = weight_variable([5*5*16, 120])
    b_fc3 = bias_variable([120])
    # shape of A_fc3 ~ [m,120]
    A_fc3 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc3) + b_fc3)
        
    ### FC4 (n = 84)
    W_fc4 = weight_variable([120, 84])
    b_fc4 = bias_variable([84])
    # shape of A_fc4 ~ [m, 84]
    A_fc4 = tf.nn.relu(tf.matmul(A_fc3, W_fc4) + b_fc4)

    # Softmax (n = 10)
    W_l = weight_variable([84, 10])
    b_l = bias_variable([10])
    # shape of A_l ~ [m,10]
    A_l=tf.nn.softmax(tf.matmul(A_fc4, W_l) + b_l)

    return A_l

''''''
#========== main process ==========#
if __name__ == "__main__":
        
    #--- load data ---#
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)  # using one-hot for output
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_valid, Y_valid = mnist.validation.images, mnist.validation.labels
    X_test,  Y_test  = mnist.test.images, mnist.test.labels
    
    #--- get the shape of data ---#
    m_train, n_x = X_train.shape
    _, n_y       = Y_train.shape
    m_valid, _   = X_valid.shape
    m_test, _    = X_test.shape
    
    #--- build the model ---#
    X = tf.placeholder(tf.float32, [None, n_x], name="X")
    Y = tf.placeholder(tf.float32, [None, n_y], name="Y")

    Y_conv = lenet_5_forward_propagation(X)
    
    # cost function
    cost = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(Y_conv, 1e-8,1.0)))
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(Y_conv,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # linked to tensorboard
    with tf.name_scope('training'):
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('accuracy', accuracy)
    
    #--- train the model ---#
    # hyper-parameter
    learning_rate = 0.001
    num_epochs = 50
    minibatch_size = 100
    
    # optimizer (using Adam)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # initial the graph and session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()  
    saver = tf.train.Saver()  # for model saving
    sess.run(init)
    
    
    
    # iterations
    start = time.time()  # time count
    train_arr=[]
    accur_train=[]
    accur_test=[]
    loss_arr=[]
    for i in range(int(num_epochs)):
        minibatch_cost=0
        for j in range( int(m_train/minibatch_size) ):
            batch_xs, batch_ys = mnist.train.next_batch(minibatch_size, shuffle = True)  # using mini-batch
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
            if(j % 100 == 0):  # print training process
                print("iteration %d training minibatch_cost %f" % (j, minibatch_cost))
        train_arr.append(i)
        loss_arr.append( minibatch_cost*100)
        accur_train.append(accuracy.eval({X: X_train, Y: Y_train})*100)
        accur_test.append( accuracy.eval({X: X_test, Y: Y_test})*100)

    end = time.time()
    print("Time consume", end-start)
    
    # save trained CNN model
    print("Parameters have been trained!")
    save_path = saver.save(sess, model_path)
    
    print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print("Valid Accuracy:", accuracy.eval({X: X_valid, Y: Y_valid}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
       
    sess.close()


    plt.figure(num='50 epochs')
    plt.subplots_adjust(hspace = 0.5)
    plt.subplot(2,1,1)
    # 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意
    plt.plot(train_arr, accur_train,color = 'b', label="Training")

    # 把資料放進來並指定對應的X軸、Y軸的資料 用圓形做標記(o-)，並指定線條顏色為綠色、使用label標記線條含意
    plt.plot(train_arr, accur_test,color = 'y', label="Testing")

    # 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
    plt.title("Accuracy")

    # 標示x軸(labelpad代表與圖片的距離)
    plt.xlabel("epoch")

    # 標示y軸(labelpad代表與圖片的距離)
    plt.ylabel("%")
    
    plt.ylim(ymax = 100)
    # 顯示出線條標記位置
    plt.legend(loc = "best")


    plt.subplot(2,1,2)
    
    plt.plot(train_arr, loss_arr,color = 'b')
   
    plt.xlabel("epoch")
    plt.ylabel("loss %")

    plt.show()
    
    print(" ~ PnYuan - PY131 ~ ")