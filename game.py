import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont
import random
import os
import sys
import time
import cProfile

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

playerSprite = Image.open(resource_path("assets/player.bmp"))
playerSpriteBW = Image.open(resource_path("assets/playerBW.bmp"))
playerBulletSprite = Image.open(resource_path("assets/playerBullet.bmp"))
enemyBulletHitSprite = Image.open(resource_path("assets/enemyBulletHit.bmp")).convert("1")
enemyBulletHitSpriteData = np.reshape(np.array(enemyBulletHitSprite.getdata(), dtype=np.uint8), (enemyBulletHitSprite.size[1], enemyBulletHitSprite.size[0]))
playerBulletHitSprite = Image.open(resource_path("assets/playerBulletHit.bmp")).convert("1")
playerBulletHitSpriteData = np.reshape(np.array(playerBulletHitSprite.getdata(), dtype=np.uint8), (playerBulletHitSprite.size[1], playerBulletHitSprite.size[0]))

playerBulletSpriteData = np.array([
    [255],
    [255],
    [255],
    [255],
])

enemyBulletSprites = []
enemyBulletSprites.append((Image.open(resource_path("assets/bullet1A.bmp")), Image.open(resource_path("assets/bullet1B.bmp")), Image.open(resource_path("assets/bullet1C.bmp")), Image.open(resource_path("assets/bullet1D.bmp"))))
enemyBulletSprites.append((Image.open(resource_path("assets/bullet2A.bmp")), Image.open(resource_path("assets/bullet2B.bmp")), Image.open(resource_path("assets/bullet2C.bmp")), Image.open(resource_path("assets/bullet2D.bmp"))))
enemyBulletSprites.append((Image.open(resource_path("assets/bullet3A.bmp")), Image.open(resource_path("assets/bullet3B.bmp")), Image.open(resource_path("assets/bullet3C.bmp")), Image.open(resource_path("assets/bullet3D.bmp"))))

enemySprites = []
enemySprites.append((Image.open(resource_path("assets/enemy1A.bmp")), Image.open(resource_path("assets/enemy1B.bmp"))))
enemySprites.append((Image.open(resource_path("assets/enemy2A.bmp")), Image.open(resource_path("assets/enemy2B.bmp"))))
enemySprites.append((Image.open(resource_path("assets/enemy3A.bmp")), Image.open(resource_path("assets/enemy3B.bmp"))))

barrierSprite = Image.open(resource_path("assets/barrier.bmp")).convert("1")


class Game:
    def __init__(self, seed=time.time()):
        self.screenRes = (256,224)

        self.stepNum = 0
        self.playerX = self.screenRes[0]//2
        self.moveAmount = 1
        self.moveDirection = 1
        self.moveIndex = 0
        self.score = 0
        self.enemies = []
        self.enemiesY = 5
        self.enemiesX = 11
        self.allEnemiesY = 0
        self.playerMoveSpeed = 2
        self.enemyBullet = {"x": 0, "y": 0, "exists": False, "type": 1, "sprite": 0}
        self.playerBullet = {"x": 0, "y": 0, "exists": False}
        self.enemyBulletSpeed = 2
        self.playerBulletSpeed = 8
        self.fireHeld = False
        self.done = False
        self.enemyBulletCooldown = 0
        self.inputs = {"left": False, "right": False, "fire": False}
        self.health = 1


        for y in reversed(range(self.enemiesY)):
            for x in range(self.enemiesX):
                enemyX = int(self.screenRes[0]/8 - 5 + (x/self.enemiesX)*(self.screenRes[0]/1.25))
                enemyY = int(self.screenRes[1]/8 + y*(self.screenRes[1]/16))
                if y == 0:
                    enemyType = 3
                    enemyTL = (4, 0)
                    enemySize = (8, 8)
                elif 0 < y < 3:
                    enemyType = 2
                    enemyTL = (3, 0)
                    enemySize = (11, 8)
                else:
                    enemyType = 1
                    enemyTL = (2, 0)
                    enemySize = (12, 8)
                    
                self.enemies.append({"x": enemyX, "y": enemyY, "TL": enemyTL, "size": enemySize, "state": 1, "type": enemyType, "sprite": 0})

        self.enemyCount = len(self.enemies)
        self.aliveEnemies = self.enemyCount

        barrierSpriteData = np.reshape(np.array(barrierSprite.getdata(), dtype=np.uint8), (barrierSprite.size[1], barrierSprite.size[0]))

        self.barriers = []
        for i in range(4):
            self.barriers.append({"x": self.screenRes[0]//5 - barrierSprite.width//2 + i*self.screenRes[0]//5, "y": 160, "data": barrierSpriteData.copy()})

    def intersects(a, asize, b, bsize):
        return not (a[0]+asize[0]-1 < b[0]
                or a[0] > b[0]+bsize[0]-1
                or a[1]+asize[1]-1 < b[1]
                or a[1] > b[1]+bsize[1]-1)

    def paste2d(arr, pos, newSize, fillValue=0):
        newArray = np.full((newSize[0]+arr.shape[0]*2, newSize[1]+arr.shape[1]*2), fillValue, dtype=np.uint8)

        newArray[pos[1]+arr.shape[0]:pos[1]+arr.shape[0]*2, pos[0]+arr.shape[1]:pos[0]+arr.shape[1]*2] = arr
        newArray = newArray[arr.shape[0]:newSize[0]+arr.shape[0], arr.shape[1]:newSize[1]+arr.shape[1]]

        return newArray

    def rand(self, start, end):
        v = random.randint(start, end)
        return v

    def step(self, inputs={"left": False, "right": False, "fire": False}):
        self.inputs = inputs
        if inputs["left"] == True and self.playerX > 24:
            self.playerX -= self.playerMoveSpeed
        if inputs["right"] == True and self.playerX < self.screenRes[0]-24:
            self.playerX += self.playerMoveSpeed
        if inputs["fire"] == True and self.fireHeld == False and self.playerBullet["exists"] == False:
            self.playerBullet["exists"] = True
            self.playerBullet["x"] = self.playerX
            self.playerBullet["y"] = 192
            self.fireHeld = True
        else:
            self.score -= abs(self.screenRes[0]/2 - self.playerX)/self.screenRes[0]/2 * 0.05
        if inputs["fire"] == False and self.fireHeld == True:
            self.fireHeld = False
        for subStep in range(self.moveAmount):
            if (self.moveIndex + 1) // self.enemyCount > 0:
                for enemy in self.enemies:
                    if enemy["state"] == 0:
                        continue
                    if not 16 < enemy["x"] < self.screenRes[0] - 32:
                        self.score -= 2
                        self.moveDirection *= -1
                        if self.moveDirection == 1:
                            self.moveAmount += 1
                        for enemy in self.enemies:
                            enemy["y"] += self.screenRes[1]//64
                            if enemy["y"] > 150 and enemy["state"] == 1:
                                self.done = True
                                self.score -= 75
                        break
            self.enemies[self.moveIndex]["x"] += self.moveDirection
            self.enemies[self.moveIndex]["sprite"] = 1 - self.enemies[self.moveIndex]["sprite"]
            self.moveIndex = (self.moveIndex + 1) % self.enemyCount

        if self.playerBullet["exists"] == True:
            if self.playerBullet["y"] < 0:
                self.playerBullet["exists"] = False

            for enemy in self.enemies:
                if enemy["state"] == 0:
                    continue
                if Game.intersects((self.playerBullet["x"], self.playerBullet["y"]+4), (1,4), (enemy["TL"][0] + enemy["x"], enemy["TL"][1] + enemy["y"]), enemy["size"]):
                    self.playerBullet["exists"] = False
                    enemy["state"] = 0
                    self.score += enemy["type"]*10
                    self.aliveEnemies -= 1
                    if self.aliveEnemies % 5 == 0:
                        self.moveAmount += 1
                    if self.aliveEnemies == 0:
                        self.done = True
                    break

            for i in range(self.playerBulletSpeed):
                if self.playerBullet["exists"] == False:
                    break
                for barrier in self.barriers:
                    if Game.intersects((self.playerBullet["x"], self.playerBullet["y"]), (1,4), (barrier["x"], barrier["y"]), (22,16)):
                        bulletMask = Game.paste2d(np.ones((4,1)), (self.playerBullet["x"]-barrier["x"], self.playerBullet["y"]-barrier["y"]), (16,22))
                        if np.any(bulletMask & barrier["data"]):
                            self.playerBullet["exists"] = False
                            mask = Game.paste2d(playerBulletHitSpriteData, (self.playerBullet["x"]-barrier["x"]-3,self.playerBullet["y"]-barrier["y"]-3), (16,22), fillValue=255)
                            barrier["data"] &= mask
                            self.score -= 0.5
                            break
                self.playerBullet["y"] -= 1

        if self.enemyBullet["exists"] == True:
            if self.stepNum % 2 == 0:
                self.enemyBullet["sprite"] = (self.enemyBullet["sprite"] + 1) % 4

            if Game.intersects((self.enemyBullet["x"], self.enemyBullet["y"]), (3,7), (self.playerX-6, 195), (13,5)):
                self.health -= 1
                if self.health == 0:
                    self.done = True
                self.score -= 50
                self.enemyBullet["exists"] = False
                self.enemyBulletCooldown = self.rand(10,60)

            if self.enemyBullet["y"] > self.screenRes[1]:
                self.enemyBullet["exists"] = False
                self.enemyBulletCooldown = self.rand(10,60)

            for i in range(self.enemyBulletSpeed):
                if self.enemyBullet["exists"] == False:
                    break
                for barrier in self.barriers:
                    if Game.intersects((self.enemyBullet["x"], self.enemyBullet["y"]), (1,8), (barrier["x"], barrier["y"]), (22,16)):
                        bulletMask = Game.paste2d(np.ones((8,1)), (self.enemyBullet["x"]-barrier["x"], self.enemyBullet["y"]-barrier["y"]), (16,22))
                        if np.any(bulletMask & barrier["data"]):
                            self.enemyBullet["exists"] = False
                            mask = Game.paste2d(enemyBulletHitSpriteData, (self.enemyBullet["x"]-barrier["x"]-3,self.enemyBullet["y"]-barrier["y"]+3), (16,22), fillValue=255)
                            barrier["data"] &= mask
                            self.enemyBulletCooldown = self.rand(10,60)
                            break
                self.enemyBullet["y"] += 1
        else:
            self.enemyBulletCooldown -= 1
                    

        if self.enemyBulletCooldown <= 0 and self.enemyBullet["exists"] == False:
            bottomIndexes = [-1] * self.enemiesX
            for i,enemy in enumerate(self.enemies):
                if bottomIndexes[i % self.enemiesX] == -1 and enemy["state"] == 1:
                    bottomIndexes[i % self.enemiesX] = i

            bottomIndexesCulled = []
            for i,v in enumerate(bottomIndexes):
                if v != -1:
                    bottomIndexesCulled.append(bottomIndexes[i])
            shootIndex = random.choice(bottomIndexesCulled)
            shootEnemy = self.enemies[shootIndex]

            self.enemyBullet["exists"] = True
            self.enemyBullet["x"] = shootEnemy["x"] + shootEnemy["TL"][0] + shootEnemy["size"][0]//2
            self.enemyBullet["y"] = shootEnemy["y"] + shootEnemy["TL"][1] + shootEnemy["size"][1]//2
            self.enemyBullet["type"] = shootEnemy["type"]-1

        if self.stepNum % 300 == 0:
            self.moveAmount += 1
        
        self.stepNum += 1

    def MLstep(self, action=0):
        lastScore = self.score
        if action == 0:
            inputs = {"left": False, "right": False, "fire": False}
        elif action == 1:
            inputs = {"left": True, "right": False, "fire": False}
        elif action == 2:
            inputs = {"left": False, "right": True, "fire": False}
        elif action == 3:
            inputs = {"left": False, "right": False, "fire": True}
        self.step(inputs)
        screen = self.drawToArray()
        return screen, self.score - lastScore, self.done
        
    def draw(self, episode=None, BW=False, ML=False):
        if BW:
            im = Image.new("1", self.screenRes, 0)
        else:
            im = Image.new("RGB", self.screenRes, (0,0,0))
        imdraw = ImageDraw.Draw(im)

        for enemy in self.enemies:
            if enemy["state"] == 0:
                continue
            im.paste(enemySprites[enemy["type"]-1][enemy["sprite"]], (enemy["x"], enemy["y"]))

        for i,barrier in enumerate(self.barriers):
            if BW:
                barrierColour = np.array([[[255,255,255]]], dtype=np.uint8)
            else:
                barrierColour = np.array([[[32,255,32]]], dtype=np.uint8)
            coloured = np.repeat(barrier["data"].reshape(16,22,1)&1, 3, axis=2)*barrierColour
            barrierSprite = Image.fromarray(coloured)
            im.paste(barrierSprite, (barrier["x"],barrier["y"]))

        if BW:
            im.paste(playerSpriteBW, (self.playerX - 6,192))
        else:
            im.paste(playerSprite, (self.playerX - 6,192))

        if self.playerBullet["exists"] == True:
            im.paste(playerBulletSprite, (self.playerBullet["x"],self.playerBullet["y"]))

        if self.enemyBullet["exists"] == True:
            im.paste(enemyBulletSprites[self.enemyBullet["type"]][self.enemyBullet["sprite"]], (self.enemyBullet["x"],self.enemyBullet["y"]))

        fnt = ImageFont.truetype(resource_path("assets/space_invaders.ttf"), 8)

        if self.inputs["left"] == True:
            inputText = "left "
        elif self.inputs["right"] == True:
            inputText = "right"
        elif self.inputs["fire"] == True:
            inputText = "fire "
        else:
            inputText = "     "
        
        if not ML:
            imdraw.text((self.screenRes[0]-75, self.screenRes[1]-12), f"Input: {inputText}", font=fnt, fill=(255,255,255), anchor="ls")
            imdraw.text((16, self.screenRes[1]-12), f"Score: {str(round(self.score,2)).rjust(4,'0')}", font=fnt, fill=(255,255,255), anchor="ls")
            if episode != None:
                imdraw.text((self.screenRes[0]//2, self.screenRes[1]-12), f"{episode}", font=fnt, fill=(255,255,255), anchor="ms")
        
        return im

    def drawToArray(self):
        im = self.draw(BW=True, ML=True)
        screen = np.asarray(im)
        return screen
