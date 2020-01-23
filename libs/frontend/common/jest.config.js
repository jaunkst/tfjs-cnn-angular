module.exports = {
  name: 'frontend-common',
  preset: '../../../jest.config.js',
  coverageDirectory: '../../../coverage/libs/frontend/common',
  snapshotSerializers: [
    'jest-preset-angular/AngularSnapshotSerializer.js',
    'jest-preset-angular/HTMLCommentSerializer.js'
  ]
};
