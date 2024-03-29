import React from 'react';
import * as tf from '@tensorflow/tfjs';
import SignatureCanvas from 'react-signature-canvas';
import './App.css';

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const MODEL_KEY = 'localstorage://model/digit_recognition';
const LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

class App extends React.Component {
  signatureCanvas = React.createRef();

  constructor(props) {
    super(props);

    this.state = {
      training: false,
      model: undefined,
      label: undefined,
      currentImage: undefined,
      dataList: [],
      prediction: [],
    };
  }

  normalizeXY = () => {
    const { dataList } = this.state;
    const numbers = [];
    const labels = [];

    dataList.forEach(dl => {
      const { data } = dl.data;
      const datasetBytesView = new Float32Array(data.length / 4);

      for (let i = 0; i <= data.length / 4; i++) {
        datasetBytesView[i] = data[i * 4] / 255;
      }

      numbers.push(datasetBytesView);
      labels.push(LABELS.indexOf(dl.label));
    });

    if (dataList.length) {
      const lastObject = dataList.slice(dataList.length - 1);

      const dbv = new Float32Array(lastObject.length / 4);

      for (let i = 0; i <= lastObject.length / 4; i++) {
        dbv[i] = lastObject[i * 4] / 255;
      }

      numbers.push(dbv);
      labels.push(LABELS.indexOf(lastObject.label));
    }

    const xs = tf.tensor(numbers).reshape([dataList.length + 1, 28, 28, 1]);
    const labelsTensor = tf.tensor1d(labels, 'int32');
    const ys = tf.oneHot(labelsTensor, LABELS.length).cast('float32');

    labelsTensor.dispose();

    return { xs, ys };
  }

  saveModel = async () => {
    const { model } = this.state;

    await model.save(MODEL_KEY);

    this.setState({ dataList: [] });
  }

  setup = () => {
    const IMG_WIDTH = 28;
    const IMG_HEIGHT = 28;
    const IMG_CHANNEL = 1;
    const { model } = this.state;

    model.add(tf.layers.conv2d({
      inputShape: [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL],
      kernelSize: 5,
      filters: 9,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));

    model.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;

    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));

    this.train();
  }

  train = async () => {
    const { model, training, dataList } = this.state;

    if (training || !dataList.length) {
      return;
    }

    this.setState({ training: true });

    const { xs, ys } = this.normalizeXY();

    model.compile({
      optimizer: tf.train.adam(),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    model.fit(xs, ys, {
      shuffle: true,
      validationSplit: 0.01,
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log('onEpochEnd', epoch, logs);
        },
        onBatchEnd: async (batch, logs) => {
          console.log('onBatchEnd', batch, logs);
          await tf.nextFrame();
        },
        onTrainEnd: () => {
          console.log('finished');
          this.setState({ training: false });
          this.saveModel();
        },
      },
    });
  }

  predict = () => {
    const { model } = this.state;
    const sourceData = new Float32Array(IMAGE_WIDTH * IMAGE_HEIGHT);
    const imageData = this.resizedImageData();

    for (let i = 0; i < imageData.data.length / 4; i++) {
      sourceData[i] = imageData.data[i * 4] / 255;
    }

    tf.tidy(() => {
      const testxs = tf.tensor(sourceData).reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
      const preds = model.predict(testxs);

      const index = preds.argMax(1).dataSync()[0];

      this.setState({
        currentImage: imageData,
        label: LABELS[index],
        prediction: Array.from(preds.dataSync())
      }, () => {
        document.getElementById('review_canvas').getContext('2d').putImageData(imageData, 0, 0);
        this.signatureCanvas.clear();
      });
    });
  }

  loadPresetData = () => {
    const width = 28;
    const height = 28;
    const imgs = document
      .getElementById('train_data')
      .getElementsByTagName('img');

    const loadImage = (img) => new Promise((resolve) => {
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, width, height);
        const { dataList } = this.state;

        dataList.push({ data: imageData, label: img.src.split('/').pop().split('.').shift() });

        this.setState({ dataList }, resolve);
      }
    });

    return Promise.all(Array.from(imgs).map(img => {
      const image = new Image();
      image.src = img.src;

      return loadImage(image);
    }));
  }

  resizedImageData = () => {
    const sourceCanvas = this.signatureCanvas.getCanvas();
    const destinationCanvas = document.createElement('canvas');

    destinationCanvas.width = IMAGE_WIDTH;
    destinationCanvas.height = IMAGE_HEIGHT;

    const tmpCtx = destinationCanvas.getContext('2d');

    tmpCtx.drawImage(
      sourceCanvas,
      0, 0, sourceCanvas.width, sourceCanvas.height,
      0, 0, IMAGE_WIDTH, IMAGE_HEIGHT
    );

    return tmpCtx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
  }

  correct = (label) => {
    const { dataList, currentImage } = this.state;

    dataList.push({ data: currentImage, label });

    this.setState({ dataList, label });
  }

  initModel = async () => {
    try {
      const m = await tf.loadLayersModel(MODEL_KEY);

      m.layers[0].trainable = false;

      this.setState({ model: m });
    } catch {
      await this.loadPresetData();

      this.setState({ model: tf.sequential() }, () => this.setup());
    }
  }

  downloadModel = () => {
    const { model } = this.state;

    model.save('downloads://digit_recognition');
  }

  loadModelFromFile = async () => {
    const uploadJSONInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights');
    const model = await tf.loadLayersModel(tf.io.browserFiles(
      [uploadJSONInput.files[0], uploadWeightsInput.files[0]])
    );

    model.layers[0].trainable = false;

    this.setState({ model });
  }

  componentDidMount() {
    this.initModel();
  }

  render() {
    const { dataList, label, prediction, training } = this.state;

    return (
      <div className="App">
        <div id="train_data" style={{ display: 'none' }}>
          <img src="/train_data/0.png" alt='' />
          <img src="/train_data/1.png" alt='' />
          <img src="/train_data/2.png" alt='' />
          <img src="/train_data/3.png" alt='' />
          <img src="/train_data/4.png" alt='' />
          <img src="/train_data/5.png" alt='' />
          <img src="/train_data/6.png" alt='' />
          <img src="/train_data/7.png" alt='' />
          <img src="/train_data/8.png" alt='' />
          <img src="/train_data/9.png" alt='' />
        </div>
        <div>data size: {dataList.length}</div>
        <div>
          <SignatureCanvas
            canvasProps={{
              width: 140,
              height: 140,
            }}
            maxWidth={8}
            backgroundColor="rgba(0,0,0,1)"
            penColor="rgba(255,255,255,1)"
            ref={ref => ref && (this.signatureCanvas = ref)}
          />
        </div>
        <button onClick={this.predict}>{training ? 'training' : 'predict'}</button>{' '}
        <button onClick={this.train}>{training ? 'training' : 'train'}</button>{' '}
        <button onClick={this.downloadModel}>download model</button>
        <hr />
        <input type="file" id="upload-json" />
        <input type="file" id="upload-weights" />
        <button onClick={this.loadModelFromFile}>upload</button>
        <hr />
        <div>
          <canvas id="review_canvas" width={28} height={28}></canvas>{' '}
          <label>it should be {label}</label>
        </div>
        <hr />
        <button onClick={() => this.correct('0')}>0</button>{' '}
        <button onClick={() => this.correct('1')}>1</button>{' '}
        <button onClick={() => this.correct('2')}>2</button>{' '}
        <button onClick={() => this.correct('3')}>3</button>{' '}
        <button onClick={() => this.correct('4')}>4</button>{' '}
        <button onClick={() => this.correct('5')}>5</button>{' '}
        <button onClick={() => this.correct('6')}>6</button>{' '}
        <button onClick={() => this.correct('7')}>7</button>{' '}
        <button onClick={() => this.correct('8')}>8</button>{' '}
        <button onClick={() => this.correct('9')}>9</button>{' '}
        <hr />
        <div>
          <ul>
            {
              prediction.map((p, i) => <li key={i}><label>{LABELS[i]}:</label> {(p * 100).toFixed(3)}%</li>)
            }
          </ul>
        </div>
      </div>
    );
  }
}

export default App;
