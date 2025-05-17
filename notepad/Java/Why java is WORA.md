because java uses jvm, which runs the bytecode (.class) can be executed by any machine with different architecture.
but the jvm for different architectural machines may vary.

without changing the machine code or compling the source code from scratch java bytecode is interpreted

java has both the compiler and an interpreter and a jit compiler.
**Java** has jdk which is different for the different architectural machines.
- **JAVAC**  is the compiler that jdk has.
- **JVM** is built using c/c++.
- inside JVM there is an **JIT** compiler inside.
what is an interpreter ?
	It is a computer program which translates the high level language source code into machine readable format and execute it as well.
		Basically it translates the source code line by line and execute it immediately after translating a line and this process repeats for so long.
	**Advantages**
		it is easy to find the errors line by line 
		debugging is made easy 
		dynamic execution (eval())
	**Disadvantages**
		It is slow compared to compiler
What is a compiler ? 
	it is a computer program which translates the source code completly and then it is executed
	**Advantages**
		It is faster then interpreting and executing line by line like an interpreter.
		the exe file generated can be distributed without a need for the source code
	**Disadvantages**
		It is a big challenge for debugging. it does not provide line by line error messages.
		compiled programs are often architecture specific.

What is an jit compiler
	It is a compiler inside a jvm which work along side with the interpreter, java monitors for the hotspots in the program if it finds a code part like a loop or method which is repeating often the jit compiler compiles it and stores in it the 
	cache for better performance.

Why java needs a compiler ? 
	It allows programmers to write once and run anywhere the class file.
	- It is faster than interpreter that doing all the jobs from translating and executing
Why java needs a interpreter ?
	It allows quick execution without waiting for full compilation or recompilation when it is distributed. 
	- And dynamic features like class loading can only be done at runtime.
Why java needs a jit compiler ?
	It improves the performance by optimizing hotspots at runtime.