.PHONY: install lint lint-fix test test-watch test-coverage check serve clean

install:          ## Install dependencies
	npm install

lint:             ## Run ESLint on js/
	npx eslint js/

lint-fix:         ## Auto-fix ESLint issues
	npx eslint js/ --fix

test:             ## Run Jest tests
	npx jest --verbose

test-watch:       ## Run Jest in watch mode
	npx jest --watchAll

test-coverage:    ## Run Jest with coverage report
	npx jest --coverage

check:            ## Run lint + tests
	npm run check

serve:            ## Serve on localhost:8080
	npx serve -l 8080 .

clean:            ## Remove node_modules and coverage
	rm -rf node_modules coverage

help:             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
