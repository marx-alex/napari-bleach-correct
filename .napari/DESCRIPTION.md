## Bleach correction for napari

This plugin is a python implementation of three different algorithms for bleach correction and can be used 
to correct time-lapse images that lose intensity due to photobleaching. The implementation is based on the ImageJ 
plugin Bleach Corrector by Miura et al. All methods work with 2D and 3D time series.

Napari Bleach correction is easy to use:

![Demo](../data/demo.gif)

### Ratio Method

This is the simplest method. Every pixel in a frame is multiplied by the ratio from the mean intensity of the 
first frame to that of the *i-th* frame.

Assumptions:
* the mean intensity is constant through the time-lapse
* the background fluorescence is the same for every pixel and frame

Parameters:
* Background Intensity: Must be estimated

### Exponential Curve Fitting

Drift estimation of fluorescence signal by fitting the mean intensity to an exponential curve.
The image is corrected by the decay in the normalized exponential function.

Assumptions:
* time intervals between frames are equal

Parameters:
* Exponential Curve: Bleaching can be modelled as a mono- or bi-exponential curve

### Histogram Matching

Bleaching correction by matching histograms to a reference image.
The correct pixel values can be calculated by the cumulative distribution function
of a frame and its reference frame. This method introduced by Miura et al.

Parameters:
* Reference Frame: Match the frame's histogram with the first our neighbor frame 

The Histogram Matching method using the neighbor frame as reference is a good start to correct bleaching.
All methods are described in detail in Miura et al.

## References

* Miura K. [Bleach correction ImageJ plugin for compensating the photobleaching of time-lapse sequences.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7871415/) F1000Res. 2020 Dec 21;9:1494. doi: 10.12688/f1000research.27171.1
* [Documentation of the ImageJ plugin](https://wiki.cmci.info/downloads/bleach_corrector)

<!-- This file is designed to provide you with a starting template for documenting
the functionality of your plugin. Its content will be rendered on your plugin's
napari hub page.

The sections below are given as a guide for the flow of information only, and
are in no way prescriptive. You should feel free to merge, remove, add and 
rename sections at will to make this document work best for your plugin. 

## Description

This should be a detailed description of the context of your plugin and its 
intended purpose.

If you have videos or screenshots of your plugin in action, you should include them
here as well, to make them front and center for new users. 

You should use absolute links to these assets, so that we can easily display them 
on the hub. The easiest way to include a video is to use a GIF, for example hosted
on imgur. You can then reference this GIF as an image.

![Example GIF hosted on Imgur](https://i.imgur.com/A5phCX4.gif)

Note that GIFs larger than 5MB won't be rendered by GitHub - we will however,
render them on the napari hub.

The other alternative, if you prefer to keep a video, is to use GitHub's video
embedding feature.

1. Push your `DESCRIPTION.md` to GitHub on your repository (this can also be done
as part of a Pull Request)
2. Edit `.napari/DESCRIPTION.md` **on GitHub**.
3. Drag and drop your video into its desired location. It will be uploaded and
hosted on GitHub for you, but will not be placed in your repository.
4. We will take the resolved link to the video and render it on the hub.

Here is an example of an mp4 video embedded this way.

https://user-images.githubusercontent.com/17995243/120088305-6c093380-c132-11eb-822d-620e81eb5f0e.mp4

## Intended Audience & Supported Data

This section should describe the target audience for this plugin (any knowledge,
skills and experience required), as well as a description of the types of data
supported by this plugin.

Try to make the data description as explicit as possible, so that users know the
format your plugin expects. This applies both to reader plugins reading file formats
and to function/dock widget plugins accepting layers and/or layer data.
For example, if you know your plugin only works with 3D integer data in "tyx" order,
make sure to mention this.

If you know of researchers, groups or labs using your plugin, or if it has been cited
anywhere, feel free to also include this information here.

## Quickstart

This section should go through step-by-step examples of how your plugin should be used.
Where your plugin provides multiple dock widgets or functions, you should split these
out into separate subsections for easy browsing. Include screenshots and videos
wherever possible to elucidate your descriptions. 

Ideally, this section should start with minimal examples for those who just want a
quick overview of the plugin's functionality, but you should definitely link out to
more complex and in-depth tutorials highlighting any intricacies of your plugin, and
more detailed documentation if you have it.

## Additional Install Steps (uncommon)
We will be providing installation instructions on the hub, which will be sufficient
for the majority of plugins. They will include instructions to pip install, and
to install via napari itself.

Most plugins can be installed out-of-the-box by just specifying the package requirements
over in `setup.cfg`. However, if your plugin has any more complex dependencies, or 
requires any additional preparation before (or after) installation, you should add 
this information here.

## Getting Help

This section should point users to your preferred support tools, whether this be raising
an issue on GitHub, asking a question on image.sc, or using some other method of contact.
If you distinguish between usage support and bug/feature support, you should state that
here.

## How to Cite

Many plugins may be used in the course of published (or publishable) research, as well as
during conference talks and other public facing events. If you'd like to be cited in
a particular format, or have a DOI you'd like used, you should provide that information here. -->

