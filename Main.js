/* jslint esversion:10 */

require('@tensorflow/tfjs-node');
const dfd = require('danfojs-node'); // import pandas as pd
const tf = require('@tensorflow/tfjs'); // import numpy as np
const SLR = require('ml-regression').SimpleLinearRegression; // from sklearn.linear_model import LinearRegression
const path = require("path");

(async () => {
    const dataset = await dfd.read_csv(`file://${path.join(__dirname,'./Salary_Data.csv')}`);

    let x = dataset.iloc({rows:[':'], columns:['0']}).values; // Matrix of features
    let y = dataset.iloc({rows:[':'], columns:['1']}).values;  // Depended variable vector

    // Spliting the dataset into the trainig and test set.
    let x_train = [], x_test = [], y_train = [], y_test = [];


    let x_shuff = [...x];
    let y_shuff = [...y];
    shuffleCombo(x_shuff, y_shuff);

    x_train = [...x_shuff.slice(0, Math.floor(0.8 * x.length))];
    x_test  = [...x_shuff.slice(Math.floor(0.8 * x.length))];

    y_train = [...y_shuff.slice(0, Math.floor(0.8 * y.length))];
    y_test  = [...y_shuff.slice(Math.floor(0.8 * y.length))];

    // regression
    const regressor = new SLR(tf.util.flatten(x_train), tf.util.flatten(y_train));
    y_pred = regressor.predict(tf.util.flatten(x_test));
    console.log(regressor.toString());
    y_pred = tf.reshape(y_pred,[-1,1]);
    y_test = tf.tensor(y_test);
    
    tf.concat([y_pred, y_test],1).print();

  })();


function shuffleCombo(array, array2) {
    
    if (array.length !== array2.length) {
      throw new Error(
        `Array sizes must match to be shuffled together ` +
        `First array length was ${array.length}` +
        `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let temp, temp2;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
      // Pick a random index
      index = (Math.random() * counter) | 0;
      // Decrease counter by 1
      counter--;
      // And swap the last element of each array with it
      temp = array[counter];
      temp2 = array2[counter];
      array[counter] = array[index];
      array2[counter] = array2[index];
      array[index] = temp;
      array2[index] = temp2;
    }
  }
  