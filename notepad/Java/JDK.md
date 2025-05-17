#### **Java Platform Independent** :
	Java as a whole is platform independent. That's because the bytecode that is generated after compilation is the same for every machine. It is the JVM that executes the bytecode using the resources of the system-specific hardware underhood, to run the program thus making it platform independent.

![[Pasted image 20250211101621.png]]
- **javac** : It is the compiler which compiles the .java files to .class files(high level to machine level).
- **java** : It is the interpreter which executes the .class files
- **javadoc** : It converts the java source code HTML document
- **javaw** : It is a standalone application launcher tool
- **javap** : disassembler which disassembles one or more class files
- jar : It Compress all files related to java applications into single unit

**JRE** :
- It acts as a environment for the java programs to execute.
- contains.
	- jvm
	- other libraries like (collection classes) :
		- util
		- math
		- io 
		- lang, etc
	It basically has the runtime libraries whenever the java program asks for them
	- AWT (Abstract Window Toolkit)
	- other integration libraries like : 
		- RMI (Remote Invocation Method)
		- JDBC (Java Data Base Connectivity)
		- JNDI (Java Naming and Definition Interface)

	**JVM** : 
	- It act as the interface between the java program and underlying hardware.
	- It lmports class files and libraries into the memory.
	- **contents** : It does'nt have any spedcific tool, it is mainly concerned with memory management activities. It is not installable separately comes with JRE.
	- **functionality** : It act as a memory management unit. And does maemory management activities like garbage collection. **It mainly talks with the underlying hardware for usage of resources to execute the java program**

