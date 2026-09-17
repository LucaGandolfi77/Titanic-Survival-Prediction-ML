import js from '@eslint/js';
import globals from 'globals';

export default [
  {
    files: ['**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      'no-unused-vars': 'warn',
      'no-console': 'warn',
      'no-debugger': 'error',
      'no-eval': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': 'error',
    },
  },
];
