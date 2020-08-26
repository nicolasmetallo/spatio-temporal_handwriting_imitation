# Spatio-Temporal Handwriting Imitation

![Pipeline Overview](pipeline.png)

Paper Link: https://arxiv.org/abs/2003.10593

## Abstract:

>Most people think that their handwriting is unique and cannot be imitated by machines, especially not using completely new content. 
Current cursive handwriting synthesis is visually limited or needs user interaction.
We show that subdividing the process into smaller subtasks makes it possible to imitate someone's handwriting with a high chance to be visually indistinguishable for humans.
Therefore, a given handwritten sample will be used as the target style. 
This sample is transferred to an online sequence. Then, a method for online handwriting synthesis is used to produce a new realistic-looking text primed with the online input sequence. This new text is then rendered and style-adapted to the input pen. We show the effectiveness of the pipeline by generating in- and out-of-vocabulary handwritten samples that are validated in a comprehensive user study. Additionally, we show that also a typical writer identification system can partially be fooled by the created fake handwritings.

tldr: Imitating someone's handwriting by converting it to the temporal domain and back again

## Requirements

See `requirements.txt`.

## Run the full pipeline

Before running the pipeline the trained model checkpoints have to be copied into the folder `checkpoints` from https://drive.google.com/open?id=11fc8b7QTSqL8oIjs7ddGutlRKEL8NBqh. To run the pipeline see `demo.sh`.

## Code contribution

The code was mainly produced by https://github.com/Finomnis

## External sources

-  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
-  https://github.com/sjvasquez/handwriting-synthesis

---
## How to run step by step

Use some helper functions

```python
class GenText:
    def __init__(self, PathToYaml):
        config = self.load_config(PathToYaml)
        self.fontStyles = self.populate_writing_styles(config['fontStyles'])
        self.penStyleTransfer = PenStyleTransfer()
        self.writer = GravesWriter()
        
    def load_config(self, PathToYaml):
        with open(PathToYaml) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    def get_pen_positions(self, imgPath):
        inputImg = Image.open(imgPath).convert('RGB')
        with Skeletonizer() as sk:
            skeletonBlurImg = sk.skeletonize_blurred(inputImg)
            skeletonImg = sk.skeletonize_sharp(skeletonBlurImg)
        return inputImg, sample_to_penpositions(skeletonImg)
        
    def populate_writing_styles(self, dictStyles):
        for k,v in dictStyles.items():
            image, penPosition = self.get_pen_positions(v['img_path'])
            dictStyles[k]['image'] = image
            dictStyles[k]['penPosition'] = penPosition
        return dictStyles
    
    def get_img_sizes(self, inputImg, newSkeletonBlurImg):
        orig_width, orig_height = inputImg.size
        skeleton_w, skeleton_h = newSkeletonBlurImg.size
        return {
            'orig': [orig_width, orig_height], 
            'skeleton': [skeleton_w, skeleton_h]}
    
    def resize_skeleton_image(self, imgSizes, newSkeletonBlurImg):
        orig_width, orig_height = imgSizes['orig']
        img_w, img_h = newSkeletonBlurImg.size
        background = Image.new('RGBA', 
                               (img_w, orig_height),
                               (255, 255, 255, 255),
                              )
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(newSkeletonBlurImg, offset)
        newSkeletonBlurImg = background.convert('RGB')
        return newSkeletonBlurImg
        
    def crop_output_image(self, imgSizes, outputImg):
        orig_width, orig_height = imgSizes['orig']
        skeleton_w, skeleton_h = imgSizes['skeleton']
        left = 0
        top = (orig_height - skeleton_h)/2
        right = skeleton_w
        bottom = (orig_height + skeleton_h)/2
        outputImg = outputImg.crop((left, top, right, bottom))
        return outputImg
    
    def remove_whitespace(self,img):
        pixdata = img.load()
        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    pixdata[x, y] = (255, 255, 255, 0)
        return img

    def write(self, styleNum, textOut, saveToDisk = False, showImg = False):
        fontStyle = self.fontStyles[styleNum]
        inputImg, penPositions = fontStyle['image'], fontStyle['penPosition']
        
        newPenPositions = self.writer.write(textOut, fontStyle['text'], penPositions)
        newPenPositions = align(newPenPositions, penPositions)
        newSkeletonBlurImg, newSkeletonImg = render_skeleton(newPenPositions)
        imgSizes = self.get_img_sizes(inputImg, newSkeletonBlurImg)
        newSkeletonBlurImg = self.resize_skeleton_image(imgSizes, newSkeletonBlurImg)
        
        outputImg = self.penStyleTransfer.transferStyle(newSkeletonBlurImg, inputImg)
        outputImg = self.crop_output_image(imgSizes, outputImg)
        
        if showImg:
            plt.figure('Full Pipeline', figsize=(16, 9))
            plt.subplot(3, 2, 1)
            plt.imshow(inputImg)
            plt.subplot(3, 2, 3)
            plt.imshow(inputImg)
            plt.subplot(3, 2, 5)
            plt.imshow(inputImg, cmap='binary', vmax=10)
            plotPenPositions(penPositions)
            plt.subplot(3, 2, 6)
            plt.imshow(newSkeletonImg, cmap='binary', vmax=256*10)
            plotPenPositions(newPenPositions)
            plt.subplot(3, 2, 4)
            plt.imshow(newSkeletonBlurImg)
            plt.subplot(3, 2, 2)
            plt.imshow(outputImg)
            plt.show()
            
        if saveToDisk:
            outputImg.save('output.png', 'PNG')
            
        return outputImg,newSkeletonImg,newSkeletonBlurImg
```

Initialise the class with

```python
%%capture
writeText = GenText('config.yaml')
```

And generate your synthetic text with string `'Amoxidal 500mg 10 capsules'`

```python
stime = time.time()
im, sk_im, new_sk_im = writeText.write(1, 'Amoxidal 500mg 10 capsules')
print(f'Total time: {time.time()-stime} seconds')
im
```

**That's it!** There are also some different writing styles that I placed on `writing_style` and you can choose within `writeText.write(<num>,<string>)`.