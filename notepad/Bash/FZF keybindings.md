```
if (( BASH_VERSINFO[0] < 4 )); then
# ALT-X - Paste the selected file path into the command line

if [[ "${FZF_CTRL_T_COMMAND-x}" != "" ]]; then
bind -m emacs-standard '"\ex": " \C-b\C-k \C-u`__fzf_select__`\e\C-e\er\C-a\C-y\C-h\C-e\e \C-y\ey\C-x\C-x\C-f"'
bind -m vi-command '"\ex": "\C-z\ex\C-z"'
bind -m vi-insert '"\ex": "\C-z\ex\C-z"'
fi

# ALT-Z - Paste the selected command from history into the command line
bind -m emacs-standard '"\ez": "\C-e \C-u\C-y\ey\C-u`__fzf_history__`\e\C-e\er"'
bind -m vi-command '"\ez": "\C-z\ez\C-z"'
bind -m vi-insert '"\ez": "\C-z\ez\C-z"'

else
# ALT-X - Paste the selected file path into the command line
if [[ "${FZF_CTRL_T_COMMAND-x}" != "" ]]; then
bind -m emacs-standard -x '"\ex": fzf-file-widget'
bind -m vi-command -x '"\ex": fzf-file-widget'
bind -m vi-insert -x '"\ex": fzf-file-widget'
fi

# ALT-Z - Paste the selected command from history into the command line
bind -m emacs-standard -x '"\ez": __fzf_history__'
bind -m vi-command -x '"\ez": __fzf_history__'
bind -m vi-insert -x '"\ez": __fzf_history__'
fi
```

