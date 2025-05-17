**1NF**
If a column has multiple values
```
1.create a separate table for that multivalued attribute or column and have it with the primary key of the main table
2.Or have separate rows for each multiple values instead of creating the new table.
```
How it should be :
	All should have a different names.
	The name of the attribute should be as same nature of the values that are stored.
	Attribute should have atomic values.
	order is not the matter here.
	
**2NF**
**Functional dependency** : 
	If a table has a primary key and all the other attributes are solely depends on that primary key.
**Partial dependency** :
	If a table has two primary keys and the if a attribute is dependent on one primary key ie. paritally dependent then it is said to partial dependency.
```
create a separate table with primary key and its dependent non-key attribute 
```
**3NF**
**Transitive dependency** : 
	If a non key attribute depends on the another non key attribute then it is known as transitive dependency.
```
create a separate table with one non key attribute as a primary key and its dependent as a non key attribute
```
**BCNF**
 If a column is a prime attribute and a non prime attribute depends on the prime attribute then it violates bcnf.
```
 if A->B where B is the Super key therefore it is a prime attribute then A should also be prime attribute.
create a separate table with the su
 
```

