spring framework used to create enterprise application.
offers ioc.inversion of control
**Spring boot**:
- spring boot is built on top of the spring framwork.
- Auto configured
- custom configuration with @annotations and xml config is no longer necessary.
- easy to setup 
- great for stand alone api
spring uses inversion of control :
	instead of programmer deciding the flow of the application this is all handover to spring framework ie. spring container.
	- dependency injection:
		instead of instiating the object manually like in passing to the constructor spring is injecting the object whenever it is needed.
		eg : userService
```
		userController using the userService 
		having :
		private userService userservice
		don't need for the creation the object with the new keyword
		spring inject for us.
		
```
**Beans**:
	an instance of a class managed by the spring container.

**what is spring container** (spring ioc):
	part of the core spring framework.
	responsible for managing all the beans.
	performs dependency injection.