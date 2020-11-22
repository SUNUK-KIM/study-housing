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

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getModel, loadData } from './model';
import { BostonHousingDataset, featureDescriptions } from './data';
import * as normalization from './normalization';
import * as ui from './ui';
import { train } from '@tensorflow/tfjs';
//start show visor
window.tf = tf;
window.tfvis = tfvis;

window.data;
window.model;

// 모델 훈련을 위한 일부 하이퍼 매개변수 설정
const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

// 로드된 데이터를 tensor로 변환하고 표준화된 버전을 생성
export function arraysToTensors() {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
  // 데이터의 평균 및 표준 편차 정규화
  let { dataMean, dataStd } =
    normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

  tensors.trainFeatures = normalization.normalizeTensor(
    tensors.rawTrainFeatures, dataMean, dataStd);
  tensors.testFeatures =
    normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

/**
 * 선형 회귀 모형을 빌드, 반환
 *
 * @returns {tf.Sequential} 선형 회귀 모델 명령
 */

// trainResult 값을 만들어주자
const trainResult = [];
let k;
const loop = 1000;

// 선형회귀 모형을 데이터로 뽑아야 함..
export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [bostonData.numFeatures], units: 1 }));
  model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense({ units: 1 }));

  model.summary();
  return model;
};

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [bostonData.numFeatures],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dense(
    { units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.summary();
  return model;
};


/**
 * 사람이 읽을 수 있는 현재 선형 가중치를 설명하기
 *
 * @param {Array} kernel Array of floats of length 12.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel) {
  tf.util.assert(
    kernel.length == 12,
    `kernel must be a array of length 12, got ${kernel.length}`);
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({ description: featureDescriptions[idx], value: kernel[idx] });
  }
  return outList;
}

/**
 * 모델을 컴파일하여 훈련데이터를 사용하여 훈련하고, 테스트 데이터에 대해 모델을 실행한다.
 * 각 epoch 후 UI를 업데이트하기 위해 콜백을 실행한다.
 *
 * @param {tf.Sequential} model 모델 훈련
 * @param {boolean} weightsIllustration 학습된 가중치에 대한 정보를 인쇄할지 여부
 */
export async function run(model, modelName, weightsIllustration) {
  model.compile(
    { optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError' });

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus('Starting training process...');
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
          `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`, modelName);
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ['loss', 'val_loss'])

        if (weightsIllustration) {
          model.layers[0].getWeights()[0].data().then(kernelAsArr => {
            const weightsList = describeKernelElements(kernelAsArr);
            ui.updateWeightDescription(weightsList);
          });
        }
      }
    }
  });

  ui.updateStatus('Running on test data...');
  const result = model.evaluate(
    tensors.testFeatures, tensors.testTarget, { batchSize: BATCH_SIZE });
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(
    `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
    `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
    `Test-set loss: ${testLoss.toFixed(4)}`,
    modelName);
};

export function computeBaseline() {
  const avgPrice = tensors.trainTarget.mean();
  console.log(`Average price: ${avgPrice.dataSync()}`);
  const baseline = tensors.testTarget.sub(avgPrice).square().mean();
  console.log(`Baseline loss: ${baseline.dataSync()}`);
  const baselineMsg = `Baseline loss (meanSquaredError) is ${baseline.dataSync()[0].toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
};

document.addEventListener('DOMContentLoaded', async () => {
  await bostonData.loadData();
  ui.updateStatus('Data loaded, converting to tensors');
  arraysToTensors();
  ui.updateStatus(
    'Data is now available as tensors.\n' +
    'Click a train button to begin.');
  // TODO Explain what baseline loss is. How it is being computed in this
  // Instance
  ui.updateBaselineStatus('Estimating baseline loss');
  computeBaseline();
  await ui.setup();
}, false);

//study//


async function initData() {
  window.data = await loadData();
}
async function showExamples() {
  // Get a surface
  const surface = tfvis.visor().surface({
    name: 'My First Surface',
    tab: 'Input Data'
  });
  const drawArea = surface.drawArea; // Get the examples

  const examples = data.nextTestBatch(22);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([28, 28, 1]);
    }); // Create a canvas element to render each example

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
}

document.querySelector('#show-examples').addEventListener('click', async e => showExamples());
// training data 
async function trains(model, data, fitCallbacks) {
  const BATCH_SIZE = 64;
  const trainDataSize = 500;
  const testDataSize = 100;
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(trainDataSize);
    return [d.xs.reshape([trainDataSize, 28, 28, 1]), d.labels];
  });
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(testDataSize);
    return [d.xs.reshape([testDataSize, 28, 28, 1]), d.labels];
  });
  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 200,
    shuffle: true,
    callbacks: fitCallbacks
  });
}
// traing 결과
async function watchTraining() {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'show.fitCallbacks',
    tab: '훈련과정(acc,loss)',
    styles: {
      height: '1000px'
    }
  }

  const callbacks = tfvis.show.fitCallbacks(container, metrics);
  return trains(model, data, callbacks);
}

document.querySelector('#start-training-1').addEventListener('click', () => watchTraining());
// train 기록 보기
async function showTrainingHistory() {
  const trainingHistory = await trains(model, data);
  tfvis.show.history({
    name: 'Training History',
    tab: '훈련기록'
  }, trainingHistory, ['loss', 'val_loss', 'acc', 'val_acc']);
}

document.querySelector('#start-training-2').addEventListener('click', () => showTrainingHistory());

// An array to hold training logs
const epochLogs = [];

async function customTrainingCharts() {
  const callbacks = {
    onEpochEnd: function (epoch, log) {
      const surface = {
        name: 'Custom Training Charts',
        tab: '훈련 차트'
      };
      const options = {
        xLabel: 'Epoch',
        yLabel: 'Value',
        yAxisDomain: [0, 1],
        seriesColors: ['teal', 'tomato']
      }; // Prep the data

      epochLogs.push(log);
      const acc = epochLogs.map((log, i) => ({
        x: i,
        y: log.acc
      }));
      const valAcc = epochLogs.map((log, i) => ({
        x: i,
        y: log.val_acc
      }));
      const data = {
        values: [acc, valAcc],
        // Custom names for the series
        series: ['Accuracy', 'Validation Accuracy'] // render the chart

      };
      tfvis.render.linechart(surface, data, options);
    }
  };
  return trains(model, data, callbacks);
}

document.querySelector('#start-training-3').addEventListener('click', () => customTrainingCharts());

// visor 에 추가하는 것 
async function runs(model, modelName, weightsIllustration) {
  model.compile(
    { optimizer: tf.train.sgd(LEARNING_RATE), loss: 'meanSquaredError' });

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus('Starting training process...');
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
          `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`, modelName);
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ['loss', 'val_loss'])

        if (weightsIllustration) {
          model.layers[0].getWeights()[0].data().then(kernelAsArr => {
            const weightsList = describeKernelElements(kernelAsArr);
            ui.updateWeightDescription(weightsList);
          });
        }
      }
    }
  });

  ui.updateStatus('Running on test data...');
  const result = model.evaluate(
    tensors.testFeatures, tensors.testTarget, { batchSize: BATCH_SIZE });
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(
    `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
    `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
    `Test-set loss: ${testLoss.toFixed(4)}`,
    modelName);
}

document.querySelector('#start-training-4').addEventListener('click', () => runs(model, 'twoHidden', false));

//show viso ----
function initModel() {
  window.model = getModel();
}

function setupListeners() {
  document.querySelector('#show-visor').addEventListener('click', () => {
    const visorInstance = tfvis.visor();
    if (!visorInstance.isOpen()) {
      visorInstance.toggle();
    }
  });

  document.querySelector('#make-first-surface')
    .addEventListener('click', () => {
      tfvis.visor().surface({ name: 'My First Surface', tab: 'Input Data' });
    });

  document.querySelector('#load-data').addEventListener('click', async (e) => {
    await initData();
    document.querySelector('#show-examples').disabled = false;
    document.querySelector('#start-training-1').disabled = false;
    document.querySelector('#start-training-2').disabled = false;
    document.querySelector('#start-training-3').disabled = false;
    document.querySelector('#start-training-4').disabled = false;
    e.target.disabled = true;
  });
}

document.addEventListener('DOMContentLoaded', function () {
  initModel();
  setupListeners();
});

showTable();