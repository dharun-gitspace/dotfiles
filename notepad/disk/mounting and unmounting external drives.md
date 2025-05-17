we are using external drives with the ntfs format, so download : 
```
sudo pacman -S ntfs-3g
```
edit the fstab
```
sudo nano /etc/fstab
```

```
# /dev/sdb1
/dev/sdb1               /media/others   ntfs-3g         defaults,uid=1000,gid=1000,dmask=0000,fmask=0000        0 0

# /dev/sdb2
/dev/sdb2               /media/education ntfs-3g        defaults,uid=1000,gid=1000,dmask=0000,fmask=0000        0 0

# /dev/sdb3
/dev/sdb3               /media/recovery ntfs-3g         defaults,uid=1000,gid=1000,dmask=0000,fmask=0000        0 0

# /dev/sdb4
/dev/sdb4               /media/dharunmr ntfs-3g         defaults,uid=1000,gid=1000,dmask=0000,fmask=0000        0 0

```

- UUID=XXXX-XXXX /mnt/your_drive ntfs-3g defaults,uid=1000,gid=1000,dmask=0000,fmask=0000 0 0`
    
    - Replace `XXXX-XXXX` with your actual drive UUID (find it with `lsblk -f` or `blkid`).
    - `uid=1000,gid=1000`: Sets the owner to your user (check your UID with `id -u`).
    - `dmask=0000`: Grants full directory access.
    - `fmask=0000`: Grants full file access.

Then reload the daemon
```
systemctl daemon-reload
```
Below command command tells Linux to mount all filesystems listed in `/etc/fstab`
```
sudo mount -a
```
To unmount the drives then do :
```
sudo umount /dir/
```