**var**:
1. `var` variables have no block scope, their visibility is scoped to current function, or global, if declared outside function.
2. `var` declarations are processed at function start (script start for globals).
**let**:
declaring twice triggers error.

## [Variable naming](https://javascript.info/variables#variable-naming)

There are two limitations on variable names in JavaScript:

1. The name must contain only letters, digits, or the symbols `$` and `_`.
2. The first character must not be a digit.

## [undefined vs ReferenceError](https://www.digitalocean.com/community/tutorials/understanding-hoisting-in-javascript#undefined-vs-referenceerror)[](https://www.digitalocean.com/community/tutorials/understanding-hoisting-in-javascript#undefined-vs-referenceerror)

Before we begin in earnest, let’s deliberate on a few things.

```
console.log(typeof variable); // Output: undefined
```

Copy

This brings us to our first point of note:

> In JavaScript, an undeclared variable is assigned the value undefined at execution and is also of type undefined.

Our second point is:

```
console.log(variable); // Output: ReferenceError: variable is not defined
```

Copy

> In JavaScript, a ReferenceError is thrown when trying to access a previously undeclared variable.

The behaviour of JavaScript when handling variables becomes nuanced because of hoisting. We’ll look at this in depth in subsequent sections.