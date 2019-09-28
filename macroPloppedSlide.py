#!python
import landslide.macro
import landslide.generator

class MyMacro(landslide.macro.Macro):
  def process(self, content, source=None):
    return content + '<p>plop</p>', ['plopped_slide']

g = landslide.generator.Generator(source='oneslide.md')
g.register_macro(MyMacro)
#print(g.render())
with open('oneslide.html', 'w') as f:
    f.write(g.render())
