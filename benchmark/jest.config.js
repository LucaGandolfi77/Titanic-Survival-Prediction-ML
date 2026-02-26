module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['<rootDir>/tests/**/*.test.js'],
  setupFilesAfterSetup: [],
  collectCoverageFrom: [
    'js/**/*.js',
    '!js/app.js'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'text-summary', 'lcov'],
  verbose: true
};
