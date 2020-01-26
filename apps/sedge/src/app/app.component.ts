import { Component, AfterViewInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

import * as tfvis from '@tensorflow/tfjs-vis';
import { ImageProcessorService, RxJimp } from '@sedge/frontend/common';

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
  filter as rxFilter,
  shareReplay as rxShareReplay
} from 'rxjs/operators';
import { map, complement, isNil, concat, reduce, max } from 'ramda';
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
const ORDERED_LABEL_CLASSES = ['yellow', 'red', 'green'];
const NUM_OUTPUT_CLASSES = ORDERED_LABEL_CLASSES.length;

const BATCH_SIZE = 100;
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

  public instantValidationImages$ = new BehaviorSubject([
    {
      src: 'http://localhost:4200/api/atlas/yellow-sample.png',
      predictedLabel: '',
      expectedLabelIndex: 0,
      confidence: 0
    },
    {
      src: 'http://localhost:4200/api/atlas/red-sample.png',
      predictedLabel: '',
      expectedLabelIndex: 1,
      confidence: 0
    },
    {
      src: 'http://localhost:4200/api/atlas/green-sample.png',
      predictedLabel: '',
      expectedLabelIndex: 2,
      confidence: 0
    }
  ]);

  public processedImages$ = new BehaviorSubject([]);

  public trainedModel$: Observable<any>;

  public testImageBlobUrl$ = new BehaviorSubject(null);
  public testImageTensors$ = this.testImageBlobUrl$.pipe(
    rxFilter(complement(isNil)),
    rxFlatMap((blobUrl: string) => {
      return this.blobURLToTensor(blobUrl);
    })
  );

  constructor(public imageProcessorService: ImageProcessorService) {}

  private blobURLToTensor(blobUrl: string) {
    return this.imageProcessorService.read(blobUrl).pipe(
      rxFlatMap((image: RxJimp) => {
        return image
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
  }

  private getLabelTensorsObservable(labels: number[]) {
    return of(tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_OUTPUT_CLASSES));
  }

  private getImageTensorsObservable(imageUrls: string[] = []) {
    return forkJoin(map(this.imageProcessorService.read, imageUrls)).pipe(
      rxFlatMap(images => {
        return forkJoin(
          map((image: any) => {
            const _processedImage = image.resize(
              TARGET_SHAPE_WIDTH,
              TARGET_SHAPE_HEIGHT
            );

            this.processedImages$.next(
              concat(this.processedImages$.getValue(), [
                _processedImage.getBase64Async('image/png')
              ])
            );

            return _processedImage.getImageDataAsObservable(
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
        strides: 3,
        activation: 'relu',
        kernelInitializer: 'randomUniform' // "varianceScaling"
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    /* Second layer */
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 8,
        strides: 3,
        activation: 'relu',
        kernelInitializer: 'randomNormal'
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Output layer
    model.add(tf.layers.flatten());
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES * NUM_OUTPUT_CLASSES * NUM_OUTPUT_CLASSES,
        kernelInitializer: 'randomNormal',
        useBias: true,
        activation: 'relu'
      })
    );
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES * NUM_OUTPUT_CLASSES,
        kernelInitializer: 'randomNormal',
        useBias: true,
        activation: 'relu'
      })
    );
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'randomNormal',
        useBias: true,
        activation: 'softmax'
      })
    );

    // Compile Model
    model.compile({
      optimizer: tf.train.adam(0.01),
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
        labelTensors.data().then(oneHotDataBuffer => {
          console.log({ oneHotData: Array.from(oneHotDataBuffer) });
        });
        const tensorFeatures = tf.stack(imageTensors);

        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

        return new Observable<tf.LayersModel>(sub => {
          model
            .fit(tensorFeatures, labelTensors, {
              batchSize: BATCH_SIZE,
              validationData: [tensorFeatures, labelTensors],
              epochs: 70,
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
    this.trainedModel$ = this.getTrainedModelObservable().pipe(
      rxShareReplay(1)
    );

    this.trainedModel$.subscribe(model => {
      this.instantValidationImages$
        .pipe(
          rxTake(1),
          rxFlatMap(validationImageList => {
            console.log({ validationImageList });
            return forkJoin(
              map(validationImage => {
                return this.blobURLToTensor(validationImage.src).pipe(
                  rxFlatMap(imageTensor => {
                    return from(model.predict(imageTensor).data());
                  }),
                  rxMap(prediction => {
                    return {
                      validationImage,
                      prediction
                    };
                  }),
                  rxTake(1)
                );
              }, validationImageList)
            );
          })
        )
        .subscribe(validationImagePredictions => {
          console.log({ validationImagePredictions });

          const updatedValidationImages: any = map(
            ({ validationImage, prediction }) => {
              const confidence = reduce(max, 0, prediction as any[]);
              return {
                ...validationImage,
                confidence,
                predictedLabel:
                  ORDERED_LABEL_CLASSES[this.indexOfMax(prediction)]
              };
            },
            validationImagePredictions
          );

          console.log({ updatedValidationImages });

          this.instantValidationImages$.next(updatedValidationImages);
        });
    });

    combineLatest([this.testImageTensors$, this.trainedModel$])
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

        const labelValues = ORDERED_LABEL_CLASSES;
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
