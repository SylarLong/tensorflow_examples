import React from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

const COLOR_LABEL = [
  'red',
  'green',
  'blue',
  'orange',
  'purple',
  'pink',
  'yellow',
  'cyan',
  'brown',
];
const MODEL_KEY = 'localstorage://model/color_recognition';
const RAWDATA_KEY = 'localstorage://rawdata/color_recognition';

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      training: false,
      model: undefined,
      r: 255,
      g: 255,
      b: 255,
      label: '',
      data: localStorage.getItem(RAWDATA_KEY) ?
        JSON.parse(localStorage.getItem(RAWDATA_KEY)) :
        [
          { r: 255, g: 0, b: 0, label: 'red' },
          { r: 0, g: 255, b: 0, label: 'green' },
          { r: 0, g: 0, b: 255, label: 'blue' },
        ]
    };
  }

  saveModel = async () => {
    const { data, model } = this.state;

    await model.save(MODEL_KEY);
    localStorage.setItem(RAWDATA_KEY, JSON.stringify(data));
  }

  normalizeXY = () => {
    const { data } = this.state;
    const colors = [];
    const labels = [];

    data.forEach(d => {
      colors.push([d.r, d.g, d.b]);
      labels.push(COLOR_LABEL.indexOf(d.label));
    });

    const xs = tf.tensor2d(colors);
    const labelsTensor = tf.tensor1d(labels, 'int32');
    const ys = tf.oneHot(labelsTensor, 9).cast('float32');

    labelsTensor.dispose();

    return { xs, ys };
  }

  setup = () => {
    const { model } = this.state;
    const hidden = tf.layers.dense({
      units: 16,
      inputShape: [3],
      activation: 'sigmoid'
    });
    const output = tf.layers.dense({
      units: 9,
      activation: 'softmax'
    });

    model.add(hidden);
    model.add(output);

    this.train();
  }

  train = async () => {
    const { model, training } = this.state;

    if (training) {
      return;
    }

    this.setState({ training: true });

    const { xs, ys } = this.normalizeXY();
    const LEARN_RATE = 0.1;
    const optimizer = tf.train.sgd(LEARN_RATE);

    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    await model.fit(xs, ys, {
      shuffle: true,
      validationSplit: 0.01,
      epochs: 250,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log('onEpochEnd', epoch, logs.loss.toFixed(5));
        },
        onBatchEnd: async (batch, logs) => {
          console.log('onBatchEnd', batch, logs);
          await tf.nextFrame();
        },
        onTrainEnd: () => {
          console.log('finished');
          this.setState({ training: false });
        },
      },
    });

    this.saveModel();
  }

  randomColor = async () => {
    const { model } = this.state;
    const r = Math.random() * 255;
    const g = Math.random() * 255;
    const b = Math.random() * 255;

    tf.tidy(() => {
      const input = tf.tensor2d([[r, g, b]]);
      const results = model.predict(input);
      const argMax = results.argMax(1);
      const index = argMax.dataSync()[0];
      const label = COLOR_LABEL[index];

      this.setState({ r, g, b, label });
    });
  }

  correct = (label) => {
    const { r, g, b, data } = this.state;

    data.push({ r, g, b, label });

    this.setState({ label });
  }

  initModel = async () => {
    try {
      const m = await tf.loadLayersModel(MODEL_KEY);

      this.setState({ model: m });
    } catch {
      this.setState({ model: tf.sequential() }, () => this.setup());
    }
  }

  downloadModel = () => {
    const { model } = this.state;

    model.save('downloads://color_recognition');
  }

  componentDidMount() {
    this.initModel();
  }

  render() {
    const { r, g, b, label, training } = this.state;

    return (
      <div id='container'>
        <button
          onClick={this.randomColor}
          style={{ width: '100%' }}
        >
          {training ? 'training...' : 'generate random color'}
        </button>
        <div id="color_card" style={{ background: `rgb(${r}, ${g}, ${b})` }}>
          {label}
        </div>
        <div id="button_container">
          <button onClick={() => this.correct('red')}>red</button>{' '}
          <button onClick={() => this.correct('green')}>green</button>{' '}
          <button onClick={() => this.correct('blue')}>blue</button>{' '}
          <button onClick={() => this.correct('orange')}>orange</button>{' '}
          <button onClick={() => this.correct('purple')}>purple</button>{' '}
          <button onClick={() => this.correct('pink')}>pink</button>{' '}
          <button onClick={() => this.correct('yellow')}>yellow</button>{' '}
          <button onClick={() => this.correct('cyan')}>cyan</button>{' '}
          <button onClick={() => this.correct('brown')}>brown</button>{' '}
        </div>
        <button onClick={this.train} style={{ width: '100%', marginTop: 10 }}>
          {training ? 'training...' : 'train model'}
        </button>
        <button onClick={this.downloadModel} style={{ width: '100%', marginTop: 10 }}>
          download model
        </button>
      </div>
    );
  }
}
export default App;
