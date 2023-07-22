import cv2
import os
import sys
import numpy as np
import shutil

def region_zoom(img_path, regions, pos='right'):
    """
    Zoom in region of the image, and concat with original image.
    - regions: two regions in (y, x, h, w) format
    """
    assert len(regions) == 2, 'There should have two regions to zoom'
    img = cv2.imread(img_path, 1)

    # sort region
    def sort_regions(x):
        if pos == 'right':
            x.sort(key= lambda ele: ele[0])
        else:
            x.sort(key= lambda ele: ele[1])
    sort_regions(regions)

    region_imgs = []
    colors = [(27, 13, 252), (80, 127, 255)]
    for re, co in zip(regions, colors):
        y, x, h, w = re
        cv2.rectangle(img, (x-1, y-1), (x+w, y+h), co, 2)
        region_imgs.append(img[y:y+h, x:x+w])

    if pos == 'right':
        region_imgs = [cv2.resize(x , (124, 124), interpolation=cv2.INTER_CUBIC) for x in region_imgs]
        h_space = np.ones((250, 2, 3)) * 255
        v_space = np.ones((2, 124, 3)) * 255
        region_comb = np.vstack((region_imgs[0], v_space, region_imgs[1]))
        return np.hstack((img, h_space, region_comb))
    elif pos == 'bottom':
        region_imgs = [cv2.resize(x , (99, 99), interpolation=cv2.INTER_CUBIC) for x in region_imgs]
        h_space = np.ones((99, 2, 3)) * 255
        v_space = np.ones((2, 200, 3)) * 255
        region_comb = np.hstack((region_imgs[0], h_space, region_imgs[1]))
        return np.vstack((img, v_space, region_comb))


def fig_intro():
    img_name = '33.jpg'
    regions = [(45, 85, 50, 50), (138, 10, 50, 50)]
    methods = ['gt_photo', 'gt_sketch', 'MWF', 'RSLCR', 'DGFL', 'FCN', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'SCG']
    for m in methods:
        img_path = os.path.join('../../result_all/CUFS_P2S', m, img_name)
        print(img_path)
        img_zoom = region_zoom(img_path, regions, 'bottom')
        cv2.imwrite('bottom_intro_{}_{}'.format(m, img_name), img_zoom)

    #  img_name = 'Dermot_Mulroney_00000128.png'
    #  img_name = 'Courtney_Thorne-Smith_00000187.png'
    #  methods = ['Photo', 'RSLCR', 'GAN', 'PS2MAN', 'Ours']
    #  for m in methods:
        #  img_path = os.path.join('../result_all/VGG/', m, img_name)
        #  img = cv2.imread(img_path)
        #  img = cv2.resize(img, (200, 250), cv2.INTER_CUBIC)
        #  cv2.imwrite('intro_{}_{}'.format(m, img_name), img)

def ablation_study():
    img_name = '99.jpg'
    regions = [(104, 104, 40, 40), (138, 10, 50, 50)]

    direction = 'AtoB'
    dirs = ['../dataset/CUFS/test_photos']
    dirs += [
            '../results_ablation/FaceSketchCycle_CUHK_N0_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N10_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N30_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P3_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P5_Sty0_v100/test_latest_iter5600_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P3_Sty1_v100/test_latest_iter3050_{}/output'.format(direction),
           ]

    imgs = []
    for d in dirs:
        img_path = os.path.join(d, img_name)
        img_zoom = region_zoom(img_path, regions)
        imgs.append(img_zoom)
    cv2.imwrite('ablation_{}.png'.format(direction), np.hstack(imgs).astype(np.uint8))

    direction = 'BtoA'
    dirs = ['../dataset/CUFS/test_sketches']
    dirs += [
            '../results_ablation/FaceSketchCycle_CUHK_N0_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N10_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N30_P1_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P3_Sty0_v100/test_latest_iter3050_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P5_Sty0_v100/test_latest_iter5600_{}/output'.format(direction),
            '../results_ablation/FaceSketchCycle_CUHK_N20_P3_Sty1_v100/test_latest_iter3050_{}/output'.format(direction),
           ]

    imgs = []
    for d in dirs:
        img_path = os.path.join(d, img_name)
        img_zoom = region_zoom(img_path, regions)
        imgs.append(img_zoom)
    cv2.imwrite('ablation_{}.png'.format(direction), np.hstack(imgs).astype(np.uint8))


def qual_eval():
    all_path = '/disk1/cfchen/result_all'

    img_name = '12.jpg'
    regions = [(138, 10, 50, 50), (30, 120, 50, 50)]
    methods = ['gt_photo', 'MWF', 'SSD', 'RSLCR', 'DGFL', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'FSW', 'SCG', 'gt_sketch']
    imgs = []
    for m in methods:
        img_path = os.path.join(all_path, 'CUFS_P2S', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img_zoom = region_zoom(img_path, regions, pos='right')
        imgs.append(img_zoom) 
    #  cv2.imwrite('qual_cufs_p2s_12.png', np.hstack(imgs).astype(np.uint8))
    cv2.imwrite('qual_cufs_p2s_12_right.png', np.hstack(imgs).astype(np.uint8))

    img_name = '224.jpg'
    regions = [(165, 75, 50, 50), (110, 100, 50, 50)]
    methods = ['gt_photo', 'MWF', 'SSD', 'RSLCR', 'DGFL', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'FSW', 'SCG', 'gt_sketch']
    imgs = []
    for m in methods:
        img_path = os.path.join(all_path, 'CUFS_P2S', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img_zoom = region_zoom(img_path, regions, pos='right')
        imgs.append(img_zoom) 
    #  cv2.imwrite('qual_cufs_p2s_224.png', np.hstack(imgs).astype(np.uint8))
    cv2.imwrite('qual_cufs_p2s_224_right.png', np.hstack(imgs).astype(np.uint8))

    img_name = '134.jpg'
    regions = [(40, 85, 40, 40), (100, 110, 40, 40)]
    methods = ['gt_photo', 'MWF', 'SSD', 'RSLCR', 'DGFL', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'FSW', 'SCG', 'gt_sketch']
    imgs = []
    for m in methods:
        img_path = os.path.join(all_path, 'CUFS_P2S', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img_zoom = region_zoom(img_path, regions, pos='right')
        imgs.append(img_zoom) 
    #  cv2.imwrite('qual_cufs_p2s_134.png', np.hstack(imgs).astype(np.uint8))
    cv2.imwrite('qual_cufs_p2s_134_right.png', np.hstack(imgs).astype(np.uint8))

    img_name = '103.jpg'
    regions = [(85, 30, 70, 70), (165, 70, 70, 70)]
    methods = ['gt_photo', 'MWF', 'SSD', 'RSLCR', 'DGFL', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'FSW', 'SCG_correct', 'gt_sketch']
    imgs = []
    for m in methods:
        #  img_path = os.path.join('../../result_all/CUFSF_P2S', m, img_name)
        img_path = os.path.join(all_path, 'CUFSF_P2S', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img_zoom = region_zoom(img_path, regions, pos='right')
        imgs.append(img_zoom) 
    #  cv2.imwrite('qual_cufsf_p2s_134.png', np.hstack(imgs).astype(np.uint8))
    cv2.imwrite('qual_cufsf_p2s_134_right.png', np.hstack(imgs).astype(np.uint8))

def qual_eval_s2p():
    img_name = '11.jpg'
    methods = ['gt_sketch', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'SCG', 'gt_photo']
    imgs = []
    for m in methods:
        img_path = os.path.join('../../result_all/CUFS_S2P', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img = cv2.imread(img_path)
        imgs.append(img) 
    cv2.imwrite('qual_cufs_s2p_11.png', np.hstack(imgs).astype(np.uint8))

    img_name = '141.jpg'
    methods = ['gt_sketch', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'SCG', 'gt_photo']
    imgs = []
    for m in methods:
        img_path = os.path.join('../../result_all/CUFS_S2P', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img = cv2.imread(img_path)
        imgs.append(img) 
    cv2.imwrite('qual_cufs_s2p_141.png', np.hstack(imgs).astype(np.uint8))


    img_name = '149.jpg'
    methods = ['gt_sketch', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'SCG', 'gt_photo']
    imgs = []
    for m in methods:
        img_path = os.path.join('../../result_all/CUFS_S2P', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img = cv2.imread(img_path)
        imgs.append(img) 
    cv2.imwrite('qual_cufs_s2p_149.png', np.hstack(imgs).astype(np.uint8))

    img_name = '103.jpg'
    methods = ['gt_sketch', 'Pix2Pix', 'PS2MAN', 'SCAGAN', 'KT', 'SCG', 'gt_photo']
    imgs = []
    for m in methods:
        img_path = os.path.join('../../result_all/CUFSF_S2P', m, img_name)
        #  img_zoom = region_zoom(img_path, regions, pos='bottom')
        img = cv2.imread(img_path)
        imgs.append(img) 
    cv2.imwrite('qual_cufs_s2p_103.png', np.hstack(imgs).astype(np.uint8))


def qual_eval_wild():
    img_names = ['Elon_Musk_00000159.jpg', 'Adam_Beach_00000182.jpg', 'Christina_Ricci_00000004.jpg', 'Brett_Lee_00000174.jpg', 'D.B._Sweeney_00000858.jpg']
    methods = ['Photo', 'SSD', 'RSLCR', 'Pix2Pix', 'PS2MAN', 'Cycle-GAN', 'FSW', 'SCG']
    for name in img_names:
        imgs = []
        for m in methods:
            img_path = os.path.join('../../result_all/VGG', m, name)
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')
            img = cv2.imread(img_path)
            img = cv2.resize(img, (200, 250))
            imgs.append(img) 
        cv2.imwrite('qual_wild_{}'.format(name), np.hstack(imgs).astype(np.uint8))



if __name__ == '__main__':
    #  fig_intro()
    #  ablation_study()
    qual_eval()
    #  qual_eval_s2p()
    #  qual_eval_wild()




