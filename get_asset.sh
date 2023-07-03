#!/bin/bash

# This script downloads the first asset from the latest Github release of a
# private repo. 
#
# PREREQUISITES
#
# curl, jq
#
# USAGE
#
# Set owner and repo variables inside the script, make sure you chmod +x it.
#
#     ./download.sh "--GITHUB TOKEN HERE--"
#

# Define variables
echo "---------------------------------------------------------------------"
echo "Define variables"
echo "---------------------------------------------------------------------"

owner="huylgia"
repo="MLOps"
tag="v1.1"
GITHUB_API_TOKEN=$1
GH_API="https://api.github.com"
GH_REPO="$GH_API/repos/$owner/$repo"
GH_TAG="$GH_REPO/releases/latest" # if get tag, we will replace "latest" with "tags/$tag"
AUTH="Authorization: token $GITHUB_API_TOKEN"

# Read asset name and id
echo "---------------------------------------------------------------------"
echo "Read asset name and id"
echo "---------------------------------------------------------------------"

response=$(curl -sH "$AUTH" $GH_TAG)
num_asset=$(echo $response | jq '.assets | length')

for i in $(seq 0 $[$num_asset-1])
do
    id=$(echo $response | jq '.assets['$i'] .id' |  tr -d '"')
    name=$(echo $response | jq '.assets['$i'] .name' |  tr -d '"')
    GH_ASSET="$GH_REPO/releases/assets/$id"

    # Print Details
    echo "---------------------------------------------------------------------"
    echo "Print Details"
    echo "Assets Id: $id"
    echo "Name: $name"
    echo "Assets URL: $GH_ASSET"
    echo "---------------------------------------------------------------------"

    # Downloading asset file
    echo "---------------------------------------------------------------------"
    echo "Downloading asset file"
    echo "---------------------------------------------------------------------"

    rm -rf ${name:0:-4}  
    curl -v -L -o "$name" -H "$AUTH" -H 'Accept: application/octet-stream' "$GH_ASSET"
    unzip $name && rm -rf $name
done