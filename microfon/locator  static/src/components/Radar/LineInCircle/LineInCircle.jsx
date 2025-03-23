import classes from "./LineInCircle.module.css"


const LineInCircle = (props) => {
    return (
        <div>
            <div className={classes.line360To180}></div>
            <div className={classes.line90To270}></div>
            <div className={classes.line45To225}></div>
            <div className={classes.line305To135}></div>
        </div>
    );
}

export default LineInCircle;