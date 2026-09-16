import js from '@eslint/js';
import globals from 'globals';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        G: 'readonly',
        window: 'readonly',
        document: 'readonly',
        navigator: 'readonly',
        localStorage: 'readonly',
        performance: 'readonly',
        requestAnimationFrame: 'readonly',
      setTimeout: 'readonly',
      clearInterval: 'readonly',
      Math: 'readonly',
      Date: 'readonly',
      console: 'readonly',
      lucide: 'readonly',
    },
    },
    rules: {
      'no-unused-vars': 'warn',
      'no-console': 'warn',
      'eqeqeq': 'warn',
      'no-var': 'warn',
      'prefer-const': 'warn',
      'no-redeclare': 'error',
    },
  },
];