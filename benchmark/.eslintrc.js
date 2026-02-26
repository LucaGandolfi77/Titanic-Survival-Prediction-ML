module.exports = {
  env: {
    browser: true,
    es2021: true,
    jest: true
  },
  globals: {
    // IIFE modules exposed as globals
    SystemInfo: 'readonly',
    CpuBenchmarks: 'readonly',
    GpuBenchmarks: 'readonly',
    MlBenchmarks: 'readonly',
    Results: 'readonly',
    StressTest: 'readonly',
    Capabilities: 'readonly',
    ExportUtils: 'readonly',
    App: 'readonly',
    // Libraries loaded via CDN
    Chart: 'readonly',
    // WebGPU API globals
    GPUBufferUsage: 'readonly',
    GPUShaderStage: 'readonly',
    GPUMapMode: 'readonly',
    // Node.js conditional export guard
    module: 'readonly'
  },
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'script'
  },
  rules: {
    // --- Possible Errors ---
    'no-cond-assign': ['error', 'except-parens'],
    'no-constant-condition': ['error', { checkLoops: false }],
    'no-dupe-args': 'error',
    'no-dupe-keys': 'error',
    'no-duplicate-case': 'error',
    'no-empty': ['warn', { allowEmptyCatch: true }],
    'no-extra-semi': 'error',
    'no-func-assign': 'error',
    'no-inner-declarations': 'error',
    'no-irregular-whitespace': 'error',
    'no-unreachable': 'error',
    'no-unsafe-negation': 'error',
    'use-isnan': 'error',
    'valid-typeof': 'error',

    // --- Best Practices ---
    'eqeqeq': ['warn', 'smart'],
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-wrappers': 'error',
    'no-self-assign': 'error',
    'no-self-compare': 'error',
    'no-throw-literal': 'error',
    'no-unused-expressions': ['warn', { allowShortCircuit: true, allowTernary: true }],
    'no-useless-concat': 'warn',
    'no-useless-return': 'warn',
    'radix': 'warn',

    // --- Variables ---
    'no-shadow': 'warn',
    'no-undef': 'error',
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    'no-use-before-define': ['error', { functions: false, classes: false }],

    // --- Style (light touch) ---
    'no-mixed-spaces-and-tabs': 'error',
    'no-trailing-spaces': 'warn',
    'semi': ['warn', 'always'],
    'quotes': ['warn', 'single', { avoidEscape: true, allowTemplateLiterals: true }],
    'comma-dangle': ['warn', 'never'],
    'no-multiple-empty-lines': ['warn', { max: 2, maxEOF: 1 }],
    'arrow-spacing': 'warn',
    'no-var': 'warn',
    'prefer-const': ['warn', { destructuring: 'all' }]
  },
  overrides: [
    {
      // Test files use Node.js / Jest environment
      files: ['tests/**/*.js', '**/*.test.js', '**/*.spec.js'],
      env: {
        node: true,
        jest: true
      }
    }
  ]
};
