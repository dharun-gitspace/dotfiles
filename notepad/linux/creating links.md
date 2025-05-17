two ways :
sym links and hard links

**hard links**
### **What is a Hard Link?**

A **hard link** is another name for an existing file. Unlike a **symbolic link (symlink)**, which acts as a shortcut, a hard link **directly points to the same inode (data on disk)**. This means:

- The original file and the hard link are **indistinguishable**.
- Deleting the original file does **not** affect the hard link (because both point to the same data).
- Hard links **cannot span different filesystems** or work with directories.
---
### **Creating a Hard Link**

Use the `ln` command **without** `-s`:

`ln original_file.txt hard_link.txt`

Now both `original_file.txt` and `hard_link.txt` point to the same inode.

---
### **Checking Hard Links**

Run:
`ls -li`
This will show:
`123456 -rw-r--r--  2 user user  100 Apr 2 13:00 hard_link.txt 123456 -rw-r--r--  2 user user  100 Apr 2 13:00 original_file.txt`

- The **first column** is the **inode number**. Both files share the same inode (`123456`), meaning they are **identical**.
- The **third column (`2`)** shows the number of hard links pointing to the inode.


**symlinks**
A **symlink** (symbolic link) is a special type of file that points to another file or directory. It acts like a shortcut, allowing you to access the target file/directory from a different location.

### How to Create a Symlink in Linux:

You use the `ln -s` command:

#### **Syntax:**

`ln -s <target> <link_name>`

- `<target>` → The original file or directory.
- `<link_name>` → The name of the symlink.
#### **Example 1: Creating a Symlink to a File**


`ln -s /home/user/original_file.txt /home/user/link_to_file.txt`

Now, `link_to_file.txt` will behave like `original_file.txt`.

#### **Example 2: Creating a Symlink to a Directory**

`ln -s /home/user/documents /home/user/my_docs`

Now, `my_docs` acts as a reference to `/home/user/documents`.

#### **Checking Symlinks**

To check if a file is a symlink and where it points:
`ls -l`
Symlinks will appear like this:
`lrwxrwxrwx 1 user user   20 Apr 2 12:34 my_docs -> /home/user/documents`