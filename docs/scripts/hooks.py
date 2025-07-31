import os
import shutil


def copy_interactives(config, **kwargs):
    site_dir = config['site_dir']
    src = 'interactives/static'
    dst = os.path.join(site_dir, 'interactives')

    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

        # Fix absolute paths in HTML files
        for root, dirs, files in os.walk(dst):
            for file in files:
                if file.endswith('.html'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        content = f.read()

                    # Replace absolute paths with relative ones
                    content = content.replace('href="/assets/', 'href="./assets/')
                    content = content.replace('src="/assets/', 'src="./assets/')

                    with open(filepath, 'w') as f:
                        f.write(content)
