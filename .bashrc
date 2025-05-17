# If not running interactively, don't do anything
[[ $- != *i* ]] && return

# Arch Linux logo (requires a Nerd Font)
ARCH_LOGO=""

# Custom Bash prompt
PS1="${ARCH_LOGO} [\u@\h \w]\$(parse_git_branch)\$ "

# Aliases
alias p='sudo pacman'
alias py='python'
alias vim='nvim'
alias activate='_activate () { source "$1"/bin/activate; };_activate'
alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias start='_start () { sudo systemctl start --now "$1";};_start'
alias stop='_stop () { sudo systemctl stop --now "$1";};_stop'
alias enable='_enable () { sudo systemctl enable --now "$1";};_enable'
alias disable='_disable () { sudo systemctl disable --now "$1";};_disable'
alias enable='_enable () { sudo systemctl enable --now "$1";};_enable'
alias status='_status () { sudo systemctl status "$1";};_status'
alias sync_time='timedatectl set-ntp true'
alias update='sudo pacman -Syu'
alias cls='clear && printf "\e[3J"'
# Function to display Git branch with Git logo
parse_git_branch() {
    local branch=$(git branch --show-current 2>/dev/null)
    if [[ -n $branch ]]; then
        echo -e "\033[0;33m  ($branch)\033[0m"  # Git logo with branch in yellow
    fi
}

# Add custom paths
export PATH="$HOME/bin:$PATH"
export RAPIDMINER_HOME="/home/dharun/Altair/RapidMiner/AI Studio 2025.0.0"
export TERMINAL=kitty

# Load Angular CLI autocompletion
# source <(ng completion script)

# FZF (if installed)
[ -f ~/.fzf.bash ] && source ~/.fzf.bash && source /usr/share/fzf/key-bindings.bash



# If not running interactively, don't do anything
case $- in
	*i*) ;;
	*) return ;;
esac

export NVM_DIR="$HOME/.nvm"
[ -s "/usr/share/nvm/init-nvm.sh" ] && \. "/usr/share/nvm/init-nvm.sh"
