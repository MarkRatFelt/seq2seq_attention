# seq2seq_attention

This code is an implementation of the model proposed here: https://blog.altoros.com/optical-character-recognition-using-one-shot-learning-rnn-and-tensorflow.html.

The input data has arithmetic data instead of receipt data; however, with the right dataset, it would work as well. 

The problem is a typical sequence to sequence problem, with the difference of one attention layer. 

I took the data set from here: https://github.com/drspiffy/Concurrence. 

With a receipt data, the X-Axis would be the line read by the OCR and the ground truth, the output data, as shown in the meetup. 

I didn't care about the code and just implemented it as an experimental guide. 


The first block "train" is to train the model manually and the second one "train-test" is for training and testing the model. 


I'd recommend using "train-test" as there is an "early stop criterion" to prevent overfitting. 

You have to customize the code to your needs. And you can let me a message if you need help with something or if you don't understand something. 


## Executing the code

```
python arithmetic_example.py
```

and these are the default options: 

```
ap.add_argument('-max_len', type=int, default=100)
ap.add_argument('-vocab_size', type=int, default=100000)
ap.add_argument('-batch_size', type=int, default=128)
ap.add_argument('-layer_num', type=int, default=1)
ap.add_argument('-hidden_dim', type=int, default=128)
ap.add_argument('-nb_epoch', type=int, default=100)
ap.add_argument('-mode', default='train-test')
```

Be sure you have installed the requirements. 
