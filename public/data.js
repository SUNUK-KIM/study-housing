/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const Papa = require('papaparse');


// Boston Housing data constants:
const BASE_URL =
  'https://storage.googleapis.com/sunuk_data/';

const TRAIN_FEATURES_FN = 'train-data9.csv';
const TRAIN_TARGET_FN = 'train-target9.csv';
const TEST_FEATURES_FN = 'test-data9.csv';
const TEST_TARGET_FN = 'test-target9.csv';

/**
 * Given CSV data returns an array of arrays of numbers.
 *
 * @param {Array<Object>} data Downloaded data.
 *
 * @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
 */
const parseCsv = async (data) => {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export const loadCsv = async (filename) => {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}`;

    console.log(`  * Downloading data from: ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']));
      }
    })
  });
};

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  constructor() {
    // Arrays to hold the data.
    this.trainFeatures = null;
    this.trainTarget = null;
    this.testFeatures = null;
    this.testTarget = null;
  }

  get numFeatures() {
    // If numFetures is accessed before the data is loaded, raise an error.
    if (this.trainFeatures == null) {
      throw new Error('\'loadData()\' must be called before numFeatures')
    }
    return this.trainFeatures[0].length;
  }

  /** Loads training and test data. */
  async loadData() {
    [this.trainFeatures, this.trainTarget, this.testFeatures, this.testTarget] =
      await Promise.all([
        loadCsv(TRAIN_FEATURES_FN), loadCsv(TRAIN_TARGET_FN),
        loadCsv(TEST_FEATURES_FN), loadCsv(TEST_TARGET_FN)
      ]);

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}

export const featureDescriptions = [
  'Contractyearmonth', 'Layer', 'surfacearea', 'Residentialarea',
  'Dedicatedarea', 'time', 'km',
  'fee', 'bus', 'sub', 'culture', 'sports',
];

/**
 * Shuffles data and target (maintaining alignment) using Fisher-Yates
 * algorithm.flab
 */
function shuffle(data, target) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // data:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // target:
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
};

//stat//

import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
          new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
      await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
      this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
      this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
      this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize, [this.trainImages, this.trainLabels], () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
        data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
        data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}
