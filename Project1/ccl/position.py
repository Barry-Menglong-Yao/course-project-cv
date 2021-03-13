class Position:
     def __init__(self,label,x_left,x_right,y_top,y_down):
          self.x_left=x_left
          self.y_top=y_top
          self.x_right=x_right
          self.y_down=y_down
          self.label=label
          self.w=0
          self.h=0
          self.character=""

     def to_result(self):
          return [self.x_left,self.y_top,self.w,self.h]

     def __str__(self):
          return  str(self.x_left)+" "+str(self.y_top)+" "+str(self.w)+" "+str(self.h)+" "+str(self.label)+" "+str(self.x_right)+" "+str(self.y_down)