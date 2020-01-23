import { Component, AfterViewInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

import * as tfvis from '@tensorflow/tfjs-vis';
import { ImageProcessorService } from '@sedge/frontend/common';

import {
  from,
  of,
  combineLatest,
  forkJoin,
  Observable,
  BehaviorSubject
} from 'rxjs';
import {
  map as rxMap,
  tap as rxTap,
  startWith as rxStartWith,
  take as rxTake,
  flatMap as rxFlatMap,
  filter as rxFilter
} from 'rxjs/operators';
import { map, complement, isNil } from 'ramda';
import { FormGroup, FormControl, Validators } from '@angular/forms';
import { Tensor } from '@tensorflow/tfjs';

const TARGET_SHAPE_WIDTH = 64;
const TARGET_SHAPE_HEIGHT = 64;
const TARGET_SHAPE_CHANNELS = 3;
const TARGET_SHAPE = [
  TARGET_SHAPE_WIDTH,
  TARGET_SHAPE_HEIGHT,
  TARGET_SHAPE_CHANNELS
];
const NUM_OUTPUT_CLASSES = 3;

const BATCH_SIZE = 64;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

const container = {
  name: 'Model Training',
  styles: { height: '1000px' }
};

@Component({
  selector: 'sedge-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
  @ViewChild('canvas', { static: true })
  public canvas: ElementRef<HTMLCanvasElement>;

  public prediction$ = new BehaviorSubject('upload a png...');

  public inputForm = new FormGroup({
    image: new FormControl(null, [Validators.required])
  });

  public testImageBlobUrl$ = new BehaviorSubject(null);
  public testImageTensors$ = this.testImageBlobUrl$.pipe(
    rxFilter(complement(isNil)),
    rxFlatMap((blobUrl: string) => {
      return this.imageProcessorService.read(blobUrl).pipe(
        rxFlatMap((image: any) => {
          return image
            .normalize()
            .resize(TARGET_SHAPE_WIDTH, TARGET_SHAPE_HEIGHT)
            .getImageDataAsObservable(
              'image/png',
              TARGET_SHAPE_WIDTH,
              TARGET_SHAPE_HEIGHT
            );
        }),
        rxMap((imageData: ImageData) => {
          return tf.stack([
            tf.browser.fromPixels(imageData, TARGET_SHAPE_CHANNELS)
          ]);
        })
      );
    })
  );

  constructor(public imageProcessorService: ImageProcessorService) {}

  private getLabelTensorsObservable(labels: number[]) {
    return of(tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_OUTPUT_CLASSES));
  }

  private getImageTensorsObservable(imageUrls: string[] = []) {
    return forkJoin(map(this.imageProcessorService.read, imageUrls)).pipe(
      rxFlatMap(images => {
        return forkJoin(
          map((image: any) => {
            return image
              .normalize()
              .resize(TARGET_SHAPE_WIDTH, TARGET_SHAPE_HEIGHT)
              .getImageDataAsObservable(
                'image/png',
                TARGET_SHAPE_WIDTH,
                TARGET_SHAPE_HEIGHT
              );
          }, images)
        );
      }),
      rxMap((imageDatas: ImageData[]) => {
        return map(imageData => {
          return tf.browser.fromPixels(
            imageData,
            TARGET_SHAPE_CHANNELS
          ) as Tensor;
        }, imageDatas);
      })
    );
  }

  private getCNNModelObservable() {
    const model = tf.sequential();
    // First layer
    model.add(
      tf.layers.conv2d({
        inputShape: TARGET_SHAPE,
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Second layer
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Output layer
    model.add(tf.layers.flatten());
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      })
    );

    // Compile Model
    model.compile({
      optimizer: tf.train.adam(0.4),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    // tfvis.show.modelSummary(container, model);

    return of(model);
  }

  private getTrainedModelObservable() {
    return forkJoin([
      this.getCNNModelObservable(),
      this.getImageTensorsObservable([
        'http://localhost:4200/api/atlas/yellow-sample.png',
        'http://localhost:4200/api/atlas/red-sample.png',
        'http://localhost:4200/api/atlas/green-sample.png'
      ]),
      this.getLabelTensorsObservable([0, 1, 2])
    ]).pipe(
      rxFlatMap(([model, imageTensors, labelTensors]) => {
        console.log('TFJS:', tf.getBackend());
        const tensorFeatures = tf.stack(imageTensors);

        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

        return new Observable<tf.LayersModel>(sub => {
          model
            .fit(tensorFeatures, labelTensors, {
              batchSize: BATCH_SIZE,
              validationData: [tensorFeatures, labelTensors],
              epochs: 2,
              shuffle: true,
              callbacks: fitCallbacks
            })
            .then((history: tf.History) => {
              sub.next(model);
              sub.complete();
            });
        });
      })
    );
  }

  onFileChange(event: any) {
    if (event.target.files && event.target.files.length) {
      const [file] = event.target.files;
      this.testImageBlobUrl$.next(URL.createObjectURL(file));
    }
  }

  ngAfterViewInit() {
    combineLatest([this.testImageTensors$, this.getTrainedModelObservable()])
      .pipe(
        rxFlatMap(([testImageTensors, model]) => {
          const results = model.predict(testImageTensors) as Tensor;
          tfvis.show.valuesDistribution(container, results);
          return from(results.data());
        })
      )
      .subscribe(prediction => {
        console.log({ prediction });

        // const labels = tf.tensor1d([0, 1, 2]);
        // const predictions = tf.tensor1d(prediction);
        // tfvis.metrics.confusionMatrix(labels, predictions).then(result => {
        //   console.log(JSON.stringify(result, null, 2));
        // });

        const labelValues = ['yellow', 'red', 'green'];
        this.prediction$.next(labelValues[this.indexOfMax(prediction)]);
      });
  }

  private indexOfMax(arr) {
    if (arr.length === 0) {
      return -1;
    }

    let max = arr[0];
    let maxIndex = 0;

    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        maxIndex = i;
        max = arr[i];
      }
    }

    return maxIndex;
  }
}
