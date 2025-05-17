In case you have spring-boot-actuator in your package, you should add the following

```
@EnableAutoConfiguration(exclude = {
        org.springframework.boot.autoconfigure.security.SecurityAutoConfiguration.class,
        org.springframework.boot.actuate.autoconfigure.ManagementWebSecurityAutoConfiguration.class})
```

With older Spring-boot, the class was called `ManagementSecurityAutoConfiguration`.

In newer versions this has changed to

```
@SpringBootApplication(exclude = {
        org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration.class,
        org.springframework.boot.actuate.autoconfigure.security.servlet.ManagementWebSecurityAutoConfiguration.class}
        )
```

**UPDATE**

If for reactive application you are having the same issue, you can exclude the following classes

```
@SpringBootApplication(exclude = {ReactiveSecurityAutoConfiguration.class, ReactiveManagementWebSecurityAutoConfiguration.class })
```

https://stackoverflow.com/a/27389784/27515943