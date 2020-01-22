module.exports = {
  name: 'sedge',
  preset: '../../jest.config.js',
  coverageDirectory: '../../coverage/apps/sedge',
  snapshotSerializers: [
    'jest-preset-angular/AngularSnapshotSerializer.js',
    'jest-preset-angular/HTMLCommentSerializer.js'
  ]
};
