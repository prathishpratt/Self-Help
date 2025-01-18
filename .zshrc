
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/prathish.murugan/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/prathish.murugan/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/Users/prathish.murugan/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/prathish.murugan/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/Users/prathish.murugan/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/Users/prathish.murugan/miniforge3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

eval "$(oh-my-posh init zsh --config /Users/prathish.murugan/Library/CloudStorage/OneDrive-Aicadium/Desktop/velvet.omp.json)"



# Aliases

alias home='cd ~'
alias desktop='cd ~/Desktop'
alias w = 'pwd'
alias ls='ls -G'
alias h='history'
alias c='clear'
alias ..='cd ..'
alias vnv='source venv/bin/activate'
alias jn='jupyter notebook'
