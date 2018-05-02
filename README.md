Yelp Restaurant Photo Classification with Kotlin and Deeplearning4j
===================================================================


For details see https://www.kaggle.com/c/yelp-restaurant-photo-classification

Download the data with
```
kaggle competitions download -c yelp-restaurant-photo-classification


```


References

* Other dl4j implementation (see gitter channel for discussion) https://gist.github.com/holgerbrandl/a522fa3156b05ec37fa55c117985c83a


## Remote execution

```bash
rsync --delete -avx --exclude target --exclude project ~/projects/deep_learning/kaggle_yelp_rest_pics/ brandl@talisker:~/projects/deep_learning/kaggle_yelp_rest_pics
```

```bash
screen -R yelp_kotlin
screen -R kotlin_yelp
cd ~/projects/deep_learning/kaggle_yelp_rest_pics

# http://www.gubatron.com/blog/2017/07/20/how-to-run-your-kotlin-gradle-built-app-from-the-command-line/

gradle run 2>&1 | tee yelp.$(date +'%Y%m%d').log
mailme "yelp done in $(pwd)"
```

## Misc

Data Prep
```bash
kaggle competitions download -c yelp-restaurant-photo-classification

#mkdir -p ~/projects/data/yelp-restaurant-photos
cd ~/projects/data/yelp-restaurant-photos

for tarFile in $(ls ~/.kaggle/competitions/yelp-restaurant-photo-classification/*.tgz); do tar xvf ${tarFile}; done

#rm train_photos/._*.jpg test_photos/._*.jpg
#https://stackoverflow.com/questions/11289551/argument-list-too-long-error-for-rm-cp-mv-commands
find train_photos -name "._*.jpg" -print0 | xargs -0 rm
find test_photos -name "._*.jpg" -print0 | xargs -0 rm

```





## References

https://tensorflow.rstudio.com/blog/keras-image-classification-on-small-datasets.html

* transfer learning increases accuracy from 80 to 90%