#!/bin/bash

function add_aliases() {
  # Write some useful aliases to "$HOME_DIR/.bash_aliases$
  local HOME_DIR="$1"

  echo -e "
  alias ll='ls -lah'\n
  " >> "${HOME_DIR}/.bash_aliases"
}

function improve_prompt() {
  # Change the prompt appearance and add current git branch
  local HOME_DIR="$1"
  local USER_MODE="$2"
  local USER_COLOR="$3"
  if [ -z "$USER_MODE" ]; then USER_MODE="01"; fi
  if [ -z "$USER_COLOR" ]; then USER_COLOR="33"; fi

  local USER_STR="\[\e[${USER_MODE};${USER_COLOR}m\]\u\[\e[00m\]"
  local WORKDIR_STR="\[\e[01;34m\]\w\[\e[00m\]"
  local GIT_STR="\[\e[0;35m\]\$(parse_git_branch)\[\e[00m\]"

  echo -e "
function parse_git_branch {
  local ref
  ref=\$(command git symbolic-ref HEAD 2> /dev/null) || return 0
  echo \"‹\${ref#refs/heads/}› \"
}

PS1='${USER_STR} :: ${WORKDIR_STR} ${GIT_STR}$ '
" >> "${HOME_DIR}/.bashrc"
}

function create_users() {
  local USERS="$1"
  local GROUP="$2"

  for x in $(tr "," "\n" <<<"$USERS"); do
    user_name="${x%/*}"
    user_id="${x#*/}"
    user_home="/home/${user_name}"
    useradd --create-home --uid "$user_id" --gid "$GROUP" --home-dir "$user_home" "$user_name"
    add_aliases "$user_home"
    improve_prompt "$user_home"
    echo "Added user ${user_name} with ID ${user_id}"
  done
}
