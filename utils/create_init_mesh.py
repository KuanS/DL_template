import cv2
import os
import sys
import numpy as np
import random
import open3d as o3d
from scipy.signal import savgol_filter
import tempfile
import shutil


class DummyLogger():
    debug = print
    info = print
    warning = print
    error = print
    critical = print
    fatal = print
    

class Z_FUNC():
    def __init__(self, zcenter, zbound):
        self.zcenter = zcenter
        self.zbound = zbound
        self.zperiod = abs(zbound-zcenter)
        
    def __call__(self, x):
        return self.zcenter + x*self.zperiod
    
class InitialMesh():
    def __init__(self, config=None, logger=None):
        # set config
        if config is not None:
            self.config = config
        else:
            self.config = None
        # set logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = DummyLogger()
        
        # set debug mode
        self.DBG = False
        
        # create temp folder
        self.tmpdir = tempfile.TemporaryDirectory()
        
        # gen Z LUT from standard sphere
        self._gen_Z_LUT()
    
    def _gen_Z_LUT(self):
        resol = 50
        #TODO: set resolution from config
        #resol = self.config.
        ver = np.array(o3d.geometry.TriangleMesh.create_sphere(resolution=resol).vertices)
        z = set()
        
        for v in ver:
            z.add(v[-1])
        zlist = sorted([a for a in z])
        ret = np.zeros((resol+1, 3))

        for idx, z in enumerate(zlist):
            temp = ver[ver[:, -1]==z]
            ret[idx] = temp[np.argmax(temp[:, 0])]

        self.Z_LUT = ret[:, [0, -1]] 
                
    def cal_cnt_normal(self, cnt):
        # smoothen contour, and get dx and dy respectively
        sx = savgol_filter(cnt[:, 0], window_length=51, polyorder=7, mode='wrap')
        dx = np.gradient(np.append(np.insert(sx, 0, sx[-1]), sx[0])*2)[1:-1]
        #dx = np.diff(np.append(np.insert(sx, 0, sx[-1]), sx[0])*2)[1:]
        sy = savgol_filter(cnt[:, 1], window_length=51, polyorder=7, mode='wrap')
        dy = np.gradient(np.append(np.insert(sy, 0, sy[-1]), sy[0])*2)[1:-1]
        #dy = np.diff(np.append(np.insert(sy, 0, sy[-1]), sy[0])*2)[1:]

        ccc = np.c_[dy, dx]
        
        # cal tangent, and get normal
        nvec = ccc/np.linalg.norm(ccc, axis=1, keepdims=True)
        nvec = nvec *np.array([-1, 1])
        return sx, sy, nvec
    
    def _read_img(self, fn):
        retval = cv2.imread(fn)
        if not retval is None:
            return retval
        else:
            self.logger.error('failed to read image file ... {}'.format(fn))
            return np.zeros((1, 1))
        
    def _write_img(self, img, fn):
        status = cv2.imwrite(fn, img)
        if not status:
            self.logger.warning('failed to write image out ... {}'.format(fn))
    
    def _get_cnt(self, img):
        if len(img.shape) > 2:
            img = img[:, :, 1]
        cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cnt
    
    def _cnt2npcnt(self, cnt):
        retval = np.empty((len(cnt[0]), 2))
        for idx, c in enumerate(cnt[0]):
            x, y = c[0]
            retval[idx] = x, y
        return retval    
    
    def _smooth_shape(self, img, outdir=None):        
        def openclose(x, ksize=5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize, ksize))
            retval = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, iterations=3)
            retval = cv2.morphologyEx(retval, cv2.MORPH_OPEN, kernel, iterations=3)
            return retval
        
        DBG = self.DBG and outdir
        temp_cnt = self._get_cnt(img)
        hull = cv2.convexHull(temp_cnt[0])
                
        pts = []
        for pt in hull:
            pts.append([*pt[0]])
        pts = np.array(pts)
        hull_mask = np.zeros_like(img, dtype=np.uint8)
        hull_mask = cv2.fillPoly(hull_mask, pts=[pts], color=(255, 255, 255))
        if DBG:
            cv2.imwrite('hull.png', hull_mask)
        
        concave = np.logical_xor(img, hull_mask).astype(np.uint8)*255
        temp_cnt = self._get_cnt(concave)
        concave_hull = np.zeros_like(img, dtype=np.uint8)
        for cnt in temp_cnt:
            hull = cv2.convexHull(cnt)
            pts = []
            for pt in hull:
                pts.append([*pt[0]])
            pts = np.array(pts)
            concave_hull = cv2.fillPoly(concave_hull, pts=[pts], color=(255, 255, 255))
        
        if DBG:
            cv2.imwrite('concave_hull.png', concave_hull)
        smooth = np.logical_xor(hull_mask, concave_hull).astype(np.uint8)*255
        smooth = openclose(smooth)
        if DBG:
            cv2.imwrite('smooth_raw.png', smooth)
        _, label, stats, _ = cv2.connectedComponentsWithStats(smooth[:,:])
        objid = np.argmax(stats[1:, -1])+1
        smooth = (label==objid).astype(np.uint8)*255
        
        smt_cnt = self._get_cnt(smooth)
        if DBG:
            cv2.imwrite('smooth.png', smooth)

        return smooth, smt_cnt, stats
        
    def _gen_top_pt(self, smooth, period=2):
        top = set()
        h, w = smooth.shape[:2]
        for i in range(0, h, period):
            for j in range(i%(period*2), w, 2*period):
                if smooth[i, j]:
                    top.add((j, i))
        return top
    
    def _gen_mid_pt(self, sx, sy, nvec, z_func):
        dilate_factor = 100
        #TODO: dilate_factor from config
        nvec_lookup = dict()
        mid = set()
        for i in range(len(self.Z_LUT)):
            dilate_step = self.Z_LUT[i, 0]
            z = z_func(self.Z_LUT[i, 1])
            for j in range(len(nvec)):
                x = sx[j]
                y = sy[j]
                dx, dy = nvec[j]
                new_x = x + dx*dilate_step*dilate_factor
                new_y = y + dy*dilate_step*dilate_factor
                k = (new_x, new_y, z)
                nvec_lookup[k] = nvec[j]
                mid.add(k)
        return mid, nvec_lookup
    
    def write_out_pt(self, foutname, top, mid, nvec_lookup, zbound, z_func):
        def gen_norm_mid(xy, xz):
            x, y = xy
            xy_norm = (x**2+y**2)**.5
            x1, z = xz
            if not (x and x1 and y and z):
                return 0., 0., 1.

            new_z = (z/x1)*xy_norm
            xyz_norm = (x**2+y**2+z**2)**.5
            return x/xyz_norm, y/xyz_norm, z/xyz_norm        
        
        xyscale, xyoffset = 5, 2.5
        zscale, zoffset = 5, 2.5

        zlookup = dict()
        for i in self.Z_LUT:
            k = z_func(i[1])
            zlookup[k] = i        
        
        with open(foutname, 'w') as f:
            f.write('X Y Z\n')
            for idx, i in enumerate(zbound):
                zn = (-1)**(idx+1)
                for c in top:
                    x, y = c
                    x = (x/1024.)*xyscale - xyoffset
                    y = (y/1024.)*xyscale - xyoffset
                    h = (i/1024)*zscale - zoffset
                    f.write('{} {} {} 0 0 {}\n'.format(x, y, h, zn))
            for c in mid:
                x, y, h = c
                nx, ny, nz = gen_norm_mid(nvec_lookup[c], zlookup[c[-1]])

                x = (x/1024.)*xyscale - xyoffset
                y = (y/1024.)*xyscale - xyoffset
                h = (h/1024)*zscale - zoffset
                f.write('{} {} {} {} {} {}\n'.format(x, y, h, nx, ny, nz))
        
                
                
    def __call__(self, img_fn):
        ## PREPARATION
        
        ## IMAGE PREPROCESS
        # read image
        img = self._read_img(img_fn)
        # resize image to 1024
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        # binarize image
        if len(img.shape) > 2:
            img = np.max(img, axis=2)
        img = ((img > 3) * 255).astype(np.uint8)
        
        # find contour
        # get hull
        # simplify the shape
        # get final contour
        smooth, smt_cnt = self._smooth_shape(img)
        self.smooth = smooth
        #top = self._gen_top_pt(smooth)
        
        
        # get boundary
        ulx, uly, bbw, bbh = cv2.boundingRect(smooth.copy())
        # set parameter (dimension)
        
        # gen Z function
        #TODO: zcenter from config (resize dimension * 0.5)
        step = 51
        step = (step//2)*2+1
        zcenter = 512
        zrange = (bbw+bbh)*.5
        zbound = [zcenter - (zrange/2)]
        zbound.append(zbound[0]+zrange)
        zfunc = Z_FUNC(zcenter, zbound[0])

        ## TOP and BOTTOM
        # sampling top and bottom points
        # add top and bottom points to a list
        top = self._gen_top_pt(smooth)
        self.top = top
        
        ## SIDE
        # formulate every contour point with its normal
        sx, sy, nvec = self.cal_cnt_normal(self._cnt2npcnt(smt_cnt))        
        # move out the contour point along its normal
        # assign the z value for each z step
        # add points to a list
        mid, nvec_lookup = self._gen_mid_pt(sx, sy, nvec, zfunc)
        
        ## WRITE OUT
        foutname = '{}/{}.xyz'.format(self.tmpdir.name, str(hash(img_fn))[1:])
        self.write_out_pt(foutname, top, mid, nvec_lookup, zbound, zfunc)
        
        ## GENERATE MESH
        pcd = o3d.io.read_point_cloud(foutname, format='xyzn')
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                                 depth=8,
                                                                                 width=0,
                                                                                 scale=1.1,
                                                                                 linear_fit=False)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)
        o3d.io.write_triangle_mesh("initmesh.ply", p_mesh_crop)
        self.logger.info('mesh generation finished')
        
aaa = InitialMesh()
t2 = '/home/kuan/Dataset/ShapeNetP2M/02828884/2b90bed7e9960ba11e672888e1de63dc/rendering/00.png'
aaa(t2)