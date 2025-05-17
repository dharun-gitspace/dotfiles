In Java, **non-primitive data types** (also called **reference types**) include everything that is not a primitive type (`int`, `char`, `float`, etc.). There are many non-primitive types available. Below is a comprehensive list:

---

## ✅ **List of Non-Primitive Data Types in Java**

### 1️⃣ **String (Immutable sequence of characters)**

- `String` is a special class in Java used to store text.
- It **can call methods** (`length()`, `charAt()`, etc.).

java

CopyEdit

`String text = "Hello"; System.out.println(text.length()); // ✅ Can call methods`

---

### 2️⃣ **Arrays (Fixed-size container of elements)**

- Arrays are objects in Java but **do not have instance methods**.
- Example: `int[]`, `String[]`

java

CopyEdit

`int[] numbers = {1, 2, 3}; System.out.println(numbers.length); // ✅ Property, NOT a method numbers.sort(); // ❌ ERROR: No method like sort() in an array`

---

### 3️⃣ **Wrapper Classes (Object versions of primitives)**

- Used to treat primitive values as objects.
- **Can call methods** like `compareTo()`, `valueOf()`, etc.

java

CopyEdit

`Integer num = 10; System.out.println(num.compareTo(5)); // ✅ Can call methods`

**Common Wrapper Classes:**

- `Integer`, `Double`, `Float`, `Long`, `Short`, `Byte`, `Boolean`, `Character`

---

### 4️⃣ **Classes (User-defined objects)**

- Objects of a class are non-primitive and **can call methods**.

java

CopyEdit

`class Person {     void greet() { System.out.println("Hello!"); } } Person p = new Person(); p.greet(); // ✅ Can call methods`

---

### 5️⃣ **Interfaces**

- Interfaces define behavior but cannot be instantiated directly.
- Implementing classes **can call methods**.

java

CopyEdit

`interface Animal { void sound(); } class Dog implements Animal {     public void sound() { System.out.println("Bark"); } } Dog d = new Dog(); d.sound(); // ✅ Can call methods`

---

### 6️⃣ **Abstract Classes**

- Cannot be instantiated but **can have methods**.

java

CopyEdit

`abstract class Vehicle {     abstract void start(); } class Car extends Vehicle {     void start() { System.out.println("Car started!"); } } Car myCar = new Car(); myCar.start(); // ✅ Can call methods`

---

### 7️⃣ **Enums (Fixed set of constants)**

- **Cannot directly call methods**, but you can define methods inside them.

java

CopyEdit

`enum Color {     RED, GREEN, BLUE; } Color c = Color.RED; // c.someMethod(); // ❌ ERROR: No default instance methods`

---

### 8️⃣ **Collections Framework (Lists, Sets, Maps)**

- These are part of `java.util` and **can call methods**.
- Example classes: `ArrayList`, `LinkedList`, `HashSet`, `HashMap`

java

CopyEdit

`ArrayList<Integer> list = new ArrayList<>(); list.add(10); // ✅ Can call methods`

---

### 9️⃣ **Records (Immutable data objects)**

- Introduced in Java 14, **can call methods**.

java

CopyEdit

`record Person(String name, int age) {} Person p = new Person("John", 25); System.out.println(p.name()); // ✅ Can call methods`

---

### 🔟 **Threads**

- Used for concurrency, **can call methods**.

java

CopyEdit

`Thread t = new Thread(() -> System.out.println("Running")); t.start(); // ✅ Can call methods`

---

### 1️⃣1️⃣ **Exception Classes (Throwable, Exception, RuntimeException, etc.)**

- **Can call methods** like `printStackTrace()`.

java

CopyEdit

`try {     throw new Exception("Error occurred!"); } catch (Exception e) {     e.printStackTrace(); // ✅ Can call methods }`

---

### **Final Summary**

|**Category**|**Examples**|**Can Call Methods?**|
|---|---|---|
|`String`|`"Hello"`|✅ Yes|
|Arrays|`int[]`, `String[]`|❌ No (Only fields like `.length`)|
|Wrapper Classes|`Integer`, `Double`, `Boolean`|✅ Yes|
|User-defined Classes|`Person`, `Car`|✅ Yes|
|Interfaces|`Runnable`, `List`|✅ Yes|
|Abstract Classes|`Vehicle`, `Animal`|✅ Yes|
|Enums|`Color.RED`, `Days.MONDAY`|❌ No (unless custom methods are added)|
|Collections|`ArrayList`, `HashMap`, `HashSet`|✅ Yes|
|Records|`record Person(String name, int age)`|✅ Yes|
|Threads|`Thread`, `Runnable`|✅ Yes|
|Exceptions|`Throwable`, `Exception`|✅ Yes|

---

### **Conclusion**

1. There are **many** non-primitive data types in Java.
2. **Most non-primitive types can call methods** because they are objects.
3. **Exceptions:**
    - **Arrays (`int[]`, `String[]`) do not have built-in methods** like collections.
    - **Enums cannot call methods directly unless explicitly defined.**